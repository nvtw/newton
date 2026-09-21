"""CPU impulse/work ledger on captured native support phases."""

import argparse
import json
from pathlib import Path

import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--source", type=Path, default=Path("/tmp/colibri_support_phases330.npz"))
parser.add_argument("--normal-load", choices=("legacy", "total"), default="legacy")
args = parser.parse_args()
source = args.source
a = np.load(source)
labels = a["labels"]
n = len(labels)


def phase(label):
    return {k.split(".", 1)[1]: a[k] for k in a.files if k.startswith(label + ".")}


def copies(s):
    v = np.concatenate((s["velocity"], s["angular_velocity"]), axis=1).astype(float)
    ends = s["copy_section_end"]
    for body in range(n):
        start = 0 if body == 0 else int(ends[body - 1])
        end = int(ends[body])
        if end > start:
            v[body, :3] = s["copy_velocity"][start:end].astype(float).mean(axis=0)
            v[body, 3:] = s["copy_angular_velocity"][start:end].astype(float).mean(axis=0)
    return v


reports = {}
for name in ("warm", "biased", "relax"):
    before = phase(name + "_before")
    solved = phase(name + "_solved")
    after = phase(name + "_averaged")
    u = copies(before)
    v = copies(after)
    W = np.zeros((n, 6, 6))
    M = np.zeros_like(W)
    movable = solved["inverse_mass"] > 0
    for b in range(n):
        W[b, :3, :3] = np.eye(3) * float(solved["inverse_mass"][b])
        xx, yy, zz, xy, xz, yz = solved["inverse_inertia"][b].astype(float)
        W[b, 3:, 3:] = [[xx, xy, xz], [xy, yy, yz], [xz, yz, zz]]
        if movable[b]:
            M[b] = np.linalg.inv(W[b])
    net = np.zeros((n, 6))
    components = {k: np.zeros_like(net) for k in ("contact_internal", "ground", "flower", "joint_hard", "joint_drive")}
    h = solved["headers"].view(np.int32)
    ground = []
    for col in range(int(solved["column_count"][0])):
        b0, b1 = h[1:3, col]
        external = not movable[b0] or not movable[b1]
        category = ("ground" if 0 in (b0, b1) else "flower") if external else "contact_internal"
        pid = int(solved["row_partition"][int(a["num_joints"][0]) + col])
        for k in range(h[5, col], h[5, col] + h[6, col]):
            normal = solved["lambdas"][:3, k].astype(float)
            t = solved["lambdas"][3:6, k].astype(float)
            axes = np.array([normal, t, np.cross(normal, t)])
            J0 = -np.concatenate((axes, np.cross(solved["derived"][9:12, k], axes)), axis=1)
            J1 = np.concatenate((axes, np.cross(solved["derived"][12:15, k], axes)), axis=1)
            lam = solved["impulses"][:, k].astype(float)
            delta = lam if name == "warm" else lam - before["impulses"][:, k]
            for body, J in ((b0, J0), (b1, J1)):
                components[category][body] += J.T @ delta
            if {b0, b1} == {0, 1}:
                vc = u.copy()
                for body in (b0, b1):
                    start = 0 if body == 0 else int(solved["copy_section_end"][body - 1])
                    end = int(solved["copy_section_end"][body])
                    match = np.flatnonzero(solved["copy_partition_list"][start:end] == pid)
                    if len(match):
                        slot = start + int(match[0])
                        vc[body] = np.r_[solved["copy_velocity"][slot], solved["copy_angular_velocity"][slot]]
                gc = J0 @ vc[b0] + J1 @ vc[b1]
                gv = J0 @ v[b0] + J1 @ v[b1]
                bias = float(solved["derived"][3, k])
                eff = float(solved["derived"][0, k])
                subtraction = 0.9417003989219666 * eff * bias
                load = (
                    float(np.clip(lam[0] + subtraction, 0, lam[0]))
                    if name != "relax" and args.normal_load == "legacy"
                    else float(lam[0])
                )
                ground.append(
                    {
                        "point": int(k),
                        "column": col,
                        "partition": pid,
                        "bias_m_s": bias,
                        "lambda_n_Ns": float(lam[0]),
                        "lambda_t_Ns": lam[1:].tolist(),
                        "estimated_bias_part_Ns": -subtraction,
                        "friction_load_Ns": load,
                        "mu": float(solved["headers"][3, col]),
                        "copy_relative_velocity": gc.tolist(),
                        "physical_relative_velocity": gv.tolist(),
                        "copy_tangent_speed_m_s": float(np.linalg.norm(gc[1:])),
                        "physical_tangent_speed_m_s": float(np.linalg.norm(gv[1:])),
                    }
                )
    jd = solved["joint_data"].view(np.int32)
    for j in range(int(a["num_joints"][0])):
        b0, b1 = jd[1:3, j]
        structural = solved["joint_structural_index"][j]
        for i in range(int(solved["joint_row_count"][j])):
            row = solved["joint_row_indices"][j, i]
            local = solved["joint_row_local"][row]
            delta = float(solved["joint_accumulated"][row]) - float(before["joint_accumulated"][row])
            category = "joint_drive" if solved["joint_row_dynamic"][row] else "joint_hard"
            for body, key in ((b0, "joint_wrench0"), (b1, "joint_wrench1")):
                components[category][body] += solved[key][structural, local].astype(float) * delta
    net = sum(components.values())
    predicted = u + np.einsum("nij,nj->ni", W, net)
    actual_impulse = np.einsum("nij,nj->ni", M, v - u)
    error = actual_impulse - net

    def PL(value, movable=movable, solved=solved):
        f = value[movable, :3]
        t = value[movable, 3:]
        x = solved["position"][movable]
        return np.r_[f.sum(axis=0), (t + np.cross(x, f)).sum(axis=0)]

    work = {key: float(np.sum(value[movable] * (u[movable] + v[movable]) * 0.5)) for key, value in components.items()}
    dE = 0.5 * float(np.einsum("ni,nij,nj->", v, M, v) - np.einsum("ni,nij,nj->", u, M, u))
    reports[name] = {
        "base_before_twist": u[1].tolist(),
        "base_after_twist": v[1].tolist(),
        "max_velocity_reconstruction_error": float(np.max(np.abs(predicted[movable] - v[movable]))),
        "momentum_defect_Ns_Nms": PL(error).tolist(),
        "external_ground_impulse": PL(components["ground"]).tolist(),
        "base_component_impulses_Ns_Nms": {k: value[1].tolist() for k, value in components.items()},
        "external_flower_impulse": PL(components["flower"]).tolist(),
        "internal_contact_momentum_defect": PL(components["contact_internal"]).tolist(),
        "joint_momentum_defect": PL(components["joint_hard"] + components["joint_drive"]).tolist(),
        "kinetic_change_J": dE,
        "phase_net_work_J": work,
        "work_balance_error_J": dE - sum(work.values()),
        "ground_points": ground,
    }
report = {
    "source": str(source),
    "normal_load_law": args.normal_load,
    "scope": "Last substep; per-phase impulse ledger at fixed pose. Work uses global phase midpoint velocity, exact additive net work attribution, not individual contact-update dissipation proof. Warm impulse is full lambda; other phases delta lambda. No copy-count rescaling of physical lambda.",
    "phases": reports,
}
source.with_suffix(".audit.json").write_text(json.dumps(report, indent=2))
print(json.dumps({k: {x: y for x, y in v.items() if x != "ground_points"} for k, v in reports.items()}, indent=2))
