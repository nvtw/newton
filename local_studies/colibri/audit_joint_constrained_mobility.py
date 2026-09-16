"""Audit a geometry-consistent joint-constrained contact mobility."""

import itertools
import json
from pathlib import Path

import numpy as np

p = np.load("/tmp/colibri_joint_reformed_input.npz")
m = np.load("/tmp/colibri_joint_constrained_mobility.npz")
s = np.load("/tmp/colibri_support_relax330.npz")
correction = np.load("/tmp/colibri_joint_constrained.npz")
active = m["active"]
G = m["G"]
slack = m["slack"]
R = p["compliance"]
drives = m["drive_rows"]
J = p["J"][:, active, :].reshape(188, -1)
M = np.zeros((len(active) * 6, len(active) * 6))
for k, b in enumerate(active):
    M[6 * k : 6 * k + 6, 6 * k : 6 * k + 6] = np.linalg.inv(p["W"][b])
rng = np.random.default_rng(483)
forces = rng.normal(size=(G.shape[0], 32))
coords = G.T @ forces
response = G @ coords
drive_delta = (slack @ coords) / np.sqrt(R[drives])[:, None]
jointres = J @ response
jointres[drives] += R[drives, None] * drive_delta
work = np.sum(forces * response, axis=0)
kin = np.sum(response * (M @ response), axis=0)
compliant = np.sum(R[drives, None] * drive_delta**2, axis=0)
out = {
    "complement_columns": G.shape[1],
    "random_min_quadratic_work": float(work.min()),
    "random_max_joint_residual": float(abs(jointres).max()),
    "max_relative_virtual_work_error": float(np.max(abs(work - kin - compliant) / np.maximum(abs(work), 1))),
    "scope": "Frozen mobility and selected actual mu-zero contacts; no live correction",
}
h = s["headers"].view(np.int32)
rows = []
ids = []
old = []
for col in range(int(s["column_count"][0])):
    if list(h[1:3, col]) != [9, 10] or s["headers"][4, col] != 0:
        continue
    for point in range(h[5, col], h[5, col] + h[6, col]):
        if s["derived"][3, point] > 0:
            continue
        n = s["lambdas"][:3, point].astype(float)
        a, b = h[1:3, col]
        c = np.zeros((38, 6))
        c[a] = -np.concatenate((n, np.cross(s["derived"][9:12, point].astype(float), n)))
        c[b] = np.concatenate((n, np.cross(s["derived"][12:15, point].astype(float), n)))
        rows.append(c[active].ravel())
        ids.append(point)
        old.append(float(s["after_impulses"][0, point]))
C = np.array(rows)
old = np.array(old)
assert 1 <= len(C) <= 8
CG = C @ G
H = CG @ CG.T
base = p["initial"][active].ravel() + correction["dv"].ravel()
rhs = C @ base - H @ old
accepted = []
for bits in itertools.product([False, True], repeat=len(C)):
    free = np.flatnonzero(bits)
    lam = np.zeros(len(C))
    if len(free):
        try:
            lam[free] = np.linalg.solve(H[np.ix_(free, free)], -rhs[free])
        except np.linalg.LinAlgError:
            continue
    gap = H @ lam + rhs
    error = max(float(np.max(-lam)), float(np.max(-gap)), float(np.max(abs(np.minimum(lam, gap)))))
    if error < 1e-8:
        accepted.append((error, lam, gap))
out["contact_points"] = ids
out["contact_mobility_eigenvalues"] = np.linalg.eigvalsh(H).tolist()
out["contact_mobility_symmetry_error"] = float(abs(H - H.T).max())
out["accepted_mode_count"] = len(accepted)
if accepted:
    error, lam, gap = min(accepted, key=lambda x: x[0])
    f = C.T @ (lam - old)
    y = G.T @ f
    dv = G @ y
    dl = (slack @ y) / np.sqrt(R[drives])
    after = base + dv
    joint = J @ dv
    joint[drives] += R[drives] * dl
    physical = (M @ dv).reshape(-1, 6)
    torque = np.sum(physical[:, 3:] + np.cross(s["position"][active].astype(float), physical[:, :3]), axis=0)
    energy = float((after @ M @ after - base @ M @ base) * 0.5)
    midpoint = (base + after) * 0.5
    contactwork = float(f @ midpoint)
    drivework = float(dl @ (J[drives] @ midpoint))
    out["accepted_subset"] = {
        "lcp_error": error,
        "normal_impulses": lam.tolist(),
        "normal_velocities": gap.tolist(),
        "joint_response_residual": float(abs(joint).max()),
        "deltaP": physical[:, :3].sum(axis=0).tolist(),
        "deltaL": torque.tolist(),
        "kinetic_energy_change": energy,
        "contact_midpoint_work": contactwork,
        "drive_midpoint_work": drivework,
        "work_balance_error": abs(energy - contactwork - drivework),
    }
np.savez("/tmp/colibri_constrained_contact_mobility.npz", H=H, C=C, G=G, old=old, rhs=rhs, base=base, active=active)
Path("/tmp/colibri_constrained_contact_mobility.json").write_text(json.dumps(out, indent=2))
print(json.dumps(out, indent=2))
