"""CPU joint/contact residuals on the two-body native phase captures."""

import json
from pathlib import Path

import numpy as np


def main():
    """Compare copied-row and averaged physical residuals with actual ownership."""
    reports = {}
    for law in ("original", "totalnormal"):
        source = "/tmp/colibri_base_frame_" + law + "_phases330.npz"
        x = np.load(source)
        phases = {}
        for phase in ("biased", "relax"):
            d = {k.split(".", 1)[1]: x[k] for k in x.files if k.startswith(phase + "_solved.")}
            ends = d["copy_section_end"]
            counts = d["copy_count"]
            mean = np.c_[d["velocity"], d["angular_velocity"]].astype(float)
            for body in range(1, len(mean)):
                start, end = int(ends[body - 1]), int(ends[body])
                if end > start:
                    mean[body] = np.r_[
                        d["copy_velocity"][start:end].astype(float).mean(0),
                        d["copy_angular_velocity"][start:end].astype(float).mean(0),
                    ]

            def slot_velocity(body, partition, ends=ends, d=d, mean=mean):
                start, end = (0 if body == 0 else int(ends[body - 1])), int(ends[body])
                found = np.flatnonzero(d["copy_partition_list"][start:end] == partition)
                if not len(found):
                    return mean[body]
                slot = start + int(found[0])
                return np.r_[d["copy_velocity"][slot], d["copy_angular_velocity"][slot]].astype(float)

            jd = d["joint_data"].view(np.int32)
            joint = []
            for j in range(int(x["num_joints"][0])):
                st = d["joint_structural_index"][j]
                pid = d["row_partition"][j]
                for row in d["joint_row_indices"][j, : d["joint_row_count"][j]]:
                    local = d["joint_row_local"][row]
                    values = []
                    for copied in (True, False):
                        value = 0.0
                        for b, key in zip(jd[1:3, j], ("joint_wrench0", "joint_wrench1"), strict=True):
                            v = slot_velocity(b, pid) if copied else mean[b]
                            value += float(d[key][st, local] @ v)
                        if d["joint_row_dynamic"][row]:
                            value += float(
                                d["joint_accumulated"][row] / d["joint_dynamic_mass"][row] - d["joint_reference"][row]
                            )
                        elif phase == "biased":
                            value += float(d["joint_bias"][st, local])
                        values.append(value)
                    joint.append(
                        {
                            "row": int(row),
                            "dynamic": bool(d["joint_row_dynamic"][row]),
                            "copy_residual": values[0],
                            "physical_residual": values[1],
                        }
                    )
            h = d["headers"].view(np.int32)
            contacts = []
            for col in range(int(d["column_count"][0])):
                b0, b1 = h[1:3, col]
                pid = int(d["row_partition"][int(x["num_joints"][0]) + col])
                for p in range(h[5, col], h[5, col] + h[6, col]):
                    impulse = d["impulses"][:, p].astype(float)
                    if impulse[0] <= 0:
                        continue
                    n, t = d["lambdas"][:3, p].astype(float), d["lambdas"][3:6, p].astype(float)
                    axes = np.array([n, t, np.cross(n, t)])
                    j0 = -np.c_[axes, np.cross(d["derived"][9:12, p], axes)]
                    j1 = np.c_[axes, np.cross(d["derived"][12:15, p], axes)]
                    copied = j0 @ slot_velocity(b0, pid) + j1 @ slot_velocity(b1, pid)
                    physical = j0 @ mean[b0] + j1 @ mean[b1]
                    bias = d["derived"][3:6, p].astype(float) if phase == "biased" else np.zeros(3)
                    load = impulse[0]
                    if law == "original" and phase == "biased" and bias[0] <= 0:
                        load = np.clip(load + 0.9417003989219666 * float(d["derived"][0, p]) * bias[0], 0, load)
                    radius = float(d["headers"][3, col]) * load
                    if phase == "biased" and d["derived"][3, p] > 0.002 / float(x["dt"][0]):
                        radius = 0.0
                    contacts.append(
                        {
                            "point": int(p),
                            "skipped_speculative_relax": bool(phase == "relax" and d["derived"][3, p] > 0),
                            "bodies": [int(b0), int(b1)],
                            "color": int(d["row_color"][int(x["num_joints"][0]) + col]),
                            "partition": pid,
                            "normal_impulse": float(impulse[0]),
                            "friction_load": float(load),
                            "tangent_impulse_norm": float(np.linalg.norm(impulse[1:])),
                            "inside_disk": bool(np.linalg.norm(impulse[1:]) < radius * 0.99),
                            "copy_velocity": copied.tolist(),
                            "physical_velocity": physical.tolist(),
                            "bias": bias.tolist(),
                            "copy_tangent_equation_norm": float(np.linalg.norm(copied[1:] + bias[1:])),
                            "physical_tangent_equation_norm": float(np.linalg.norm(physical[1:] + bias[1:])),
                        }
                    )
            phases[phase] = {
                "copy_counts": counts.tolist(),
                "joint": joint,
                "contacts": contacts,
                "base_physical_twist": mean[1].tolist(),
            }
        reports[law] = {"source": source, "phases": phases}
    Path("/tmp/colibri_base_frame_phase_audit.json").write_text(json.dumps(reports, indent=2))
    for law, r in reports.items():
        for phase, p in r["phases"].items():
            print(law, phase, "joint", p["joint"])
            eligible = [c for c in p["contacts"] if c["inside_disk"] and not c["skipped_speculative_relax"]]
            print(
                "worst interior", max(eligible, key=lambda c: c["physical_tangent_equation_norm"]) if eligible else None
            )


if __name__ == "__main__":
    main()
