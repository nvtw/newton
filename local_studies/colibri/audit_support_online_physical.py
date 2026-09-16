"""Independent physical audit of the online all-point biased support solve."""

import json
import argparse
from pathlib import Path
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--input", default="/tmp/colibri_support_online_continuation.npz")
parser.add_argument("--output", default="/tmp/colibri_support_online_physical")
args = parser.parse_args()
d = np.load(args.input)
meta = json.load(open("/tmp/colibri_two_body_full_biased.json"))
src = np.load(meta["source"])
s = {k.split(".", 1)[1]: src[k] for k in src.files if k.startswith("biased_solved.")}
W, B, C = d["W"], d["B"], d["C"]
lam = d["solution"]
free = d["velocity"] - W @ (C.T @ d["initial"] + B.T @ d["old_joint"])
K = B @ W @ B.T + np.diag(d["diagonal"])
joint = np.linalg.solve(K, d["targets"] - B @ free - B @ W @ C.T @ lam)
v = free + W @ (C.T @ lam + B.T @ joint)
vbar = free + W @ B.T @ np.linalg.solve(K, d["targets"] - B @ free)
bias = d["rhs"] - C @ vbar
gradient = (C @ v + bias).reshape(-1, 3)
gradient[:, 0] += d["regularization"] * lam[::3]
l = lam.reshape(-1, 3)
radius = d["mu"] * l[:, 0]
tangent = np.linalg.norm(gradient[:, 1:], axis=1)
tangent_impulse = np.linalg.norm(l[:, 1:], axis=1)
normal_feasibility = max(float(-l[:, 0].min()), float(-gradient[:, 0].min()), 0.0)
normal_product = float(np.max(np.abs(l[:, 0] * gradient[:, 0])))
cone_violation = float(np.max(np.maximum(tangent_impulse - radius, 0)))
friction_identity = np.linalg.norm(tangent[:, None] * l[:, 1:] + radius[:, None] * gradient[:, 1:], axis=1)
friction_error = float(friction_identity.max())
joint_error = float(np.max(np.abs(B @ v + d["diagonal"] * joint - d["targets"])))
delta = lam - d["initial"]
impulse = C.T @ delta + B.T @ (joint - d["old_joint"])
momentum = np.zeros(6)
for i, b in enumerate(meta["bodies"]):
    p = impulse[6 * i : 6 * i + 6]
    momentum += np.r_[p[:3], p[3:] + np.cross(s["position"][b], p[:3])]
h = s["headers"].view(np.int32)
reaction = np.zeros(6)
for i, p in enumerate(meta["points"]):
    col = next(j for j in range(int(s["column_count"][0])) if h[5, j] <= p < h[5, j] + h[6, j])
    n, t = s["lambdas"][:3, p].astype(float), s["lambdas"][3:6, p].astype(float)
    axes = np.array([n, t, np.cross(n, t)])
    for b, sign, start in ((h[1, col], -1, 9), (h[2, col], 1, 12)):
        if b == 0:
            force = sign * (axes.T @ delta.reshape(-1, 3)[i])
            reaction += np.r_[force, np.cross(s["position"][0] + s["derived"][start : start + 3, p], force)]
momentum += reaction
mid = (v + d["velocity"]) * 0.5
work = float(mid @ impulse)
energy = float(0.5 * (v @ np.linalg.solve(W, v) - d["velocity"] @ np.linalg.solve(W, d["velocity"])))
normal = np.zeros_like(delta)
normal[::3] = delta[::3]
parts = dict(
    contact_normal=float(mid @ C.T @ normal),
    contact_tangent=float(mid @ C.T @ (delta - normal)),
    hard_joint=float(mid @ B[d["diagonal"] == 0].T @ (joint - d["old_joint"])[d["diagonal"] == 0]),
    actual_drive=float(mid @ B[d["diagonal"] > 0].T @ (joint - d["old_joint"])[d["diagonal"] > 0]),
)
gpu = np.load("/tmp/single_block_support_biased.npz")
vgpu = gpu["baseline"] + gpu["G"] @ lam
report = dict(
    all_points=len(l),
    normal_feasibility=normal_feasibility,
    normal_complementarity_product=normal_product,
    cone_violation=cone_violation,
    maximum_dissipation_identity_error=friction_error,
    joint_equation_error=joint_error,
    momentum_with_ground_reaction_error=momentum.tolist(),
    energy_change=energy,
    midpoint_work=work,
    work_error=abs(work - energy),
    work_parts=parts,
    gpu_response_velocity_error=float(np.max(np.abs(v - vgpu))),
    loaded_points=np.flatnonzero(l[:, 0] > 1e-12).tolist(),
    loaded_slip_speed=tangent[l[:, 0] > 1e-12].tolist(),
    base_before=d["velocity"][:6].tolist(),
    base_after=v[:6].tolist(),
    scope="All67 original biased contacts including speculative normal/tangent offsets; actual drive and compliance. Frozen solve, not live stationarity proof.",
)
assert normal_feasibility < 1e-8 and normal_product < 1e-12 and cone_violation < 1e-8
assert friction_error < 1e-10 and joint_error < 1e-8
assert np.max(np.abs(momentum)) < 2e-10 and abs(work - energy) < 1e-14
assert np.max(np.abs(v - vgpu)) < 1e-10
np.savez(args.output + ".npz", velocity=v, joint=joint, impulse=lam, gradient=gradient)
Path(args.output + ".json").write_text(json.dumps(report, indent=2))
print(json.dumps(report, indent=2))
