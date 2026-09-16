# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""CPU physical-space joint/soft-normal active-set reference, no live override.

Each active soft normal is eliminated exactly into the physical velocity system.
No contact row is truncated. Hard active normals are explicitly unsupported by
this first bounded prototype and cause rejection, not a hidden regularizer.
"""

import json
from pathlib import Path

import numpy as np

from .coupled_support_online import assemble_snapshot


def solve_equilibrated(matrix, rhs):
    """Positive row/column scaling; original equations remain the acceptance gate."""
    row = np.max(abs(matrix), axis=1)
    assert np.all(row > 0)
    scaled = matrix / row[:, None]
    col = np.max(abs(scaled), axis=0)
    assert np.all(col > 0)
    return np.linalg.solve(scaled / col[None, :], rhs / row) / col


def solve_normal(a, dtype=np.float32, max_solves=128, whiten=False):
    """Hold original tangential impulses fixed and solve joint+normal KKT."""
    w, b, c = [a[k].astype(dtype) for k in ("W", "B", "C")]
    cn = c[::3]
    gamma = a["gamma"].astype(dtype)
    bias = (a["rhs"] - a["C"] @ a["vbar"])[::3].astype(dtype)
    diagonal = a["diagonal"].astype(dtype)
    target = a["targets"].astype(dtype)
    old = a["old"][::3].astype(dtype)
    free = a["velocity"].astype(dtype) - w @ (b.T @ a["old_joint"].astype(dtype) + cn.T @ old)
    mass = np.linalg.inv(w)
    count = len(old)
    nv = len(free)
    nj = len(target)
    if whiten:
        root = np.linalg.cholesky(w)
        normal_rows = cn @ root
        joint_scale = dtype(1) / np.sqrt(np.einsum("ij,ji->i", b @ w, b.T) + diagonal)
        joint_rows = joint_scale[:, None] * (b @ root)
        joint_diagonal = joint_scale * diagonal * joint_scale
        normal_target = bias + cn @ free
        joint_target = joint_scale * (target - b @ free)
    else:
        root = np.eye(nv, dtype=dtype)
        normal_rows, joint_rows = cn, b
        joint_diagonal = diagonal
        normal_target, joint_target = bias, target
        joint_scale = np.ones(nj, dtype=dtype)
    active = old > 0
    current = np.maximum(old, dtype(0))
    stats = []
    for iteration in range(max_solves):
        ids = np.flatnonzero(active)
        if np.any(gamma[ids] <= 0):
            return {
                "accepted": False,
                "reason": "active hard normal requires separate certified quotient",
                "steps": stats,
            }
        h = np.eye(nv, dtype=dtype) if whiten else mass.copy()
        rhs = np.zeros(nv, dtype=dtype) if whiten else mass @ free
        if len(ids):
            scaled = normal_rows[ids] / gamma[ids, None]
            h += normal_rows[ids].T @ scaled
            rhs -= scaled.T @ normal_target[ids]
        matrix = np.zeros((nv + nj, nv + nj), dtype=dtype)
        matrix[:nv, :nv] = h
        matrix[:nv, nv:] = -joint_rows.T
        matrix[nv:, :nv] = joint_rows
        matrix[nv:, nv:] = np.diag(joint_diagonal)
        solution = solve_equilibrated(matrix, np.r_[rhs, joint_target])
        velocity = free + root @ solution[:nv] if whiten else solution[:nv]
        joint = joint_scale * solution[nv:]
        candidate = np.zeros(count, dtype=dtype)
        candidate[ids] = -(cn[ids] @ velocity + bias[ids]) / gamma[ids]
        bad = ids[candidate[ids] < 0]
        if len(bad):
            fraction = current[bad] / (current[bad] - candidate[bad])
            index = int(bad[np.argmin(fraction)])
            alpha = np.min(fraction)
            current += alpha * (candidate - current)
            current[index] = 0
            active[index] = False
            stats.append({"action": "remove", "point": index, "alpha": float(alpha)})
            continue
        current = candidate
        gradient = cn @ velocity + bias + gamma * current
        inactive = np.flatnonzero(~active)
        if len(inactive) and np.min(gradient[inactive]) < -1e-8:
            index = int(inactive[np.argmin(gradient[inactive])])
            active[index] = True
            stats.append({"action": "add", "point": index, "gradient": float(gradient[index])})
            continue
        # Independent original FP64 equations, including physical impulse scatter.
        vd = velocity.astype(float)
        jd = joint.astype(float)
        nd = current.astype(float)
        cn64 = a["C"][::3]
        b64 = (a["rhs"] - a["C"] @ a["vbar"])[::3]
        gd = cn64 @ vd + b64 + a["gamma"] * nd
        diag = np.einsum("ij,ji->i", cn64 @ a["W"], cn64.T) + a["gamma"]
        natural = diag * (nd - np.maximum(0, nd - gd / diag))
        joint_res = a["B"] @ vd + a["diagonal"] * jd - a["targets"]
        response = vd - a["velocity"] - a["W"] @ (a["B"].T @ (jd - a["old_joint"]) + cn64.T @ (nd - a["old"][::3]))
        errors = {
            "normal_natural_max": float(np.max(abs(natural))),
            "joint_max": float(np.max(abs(joint_res))),
            "response_max": float(np.max(abs(response))),
            "nonnegative": float(max(0, -np.min(nd))),
        }
        return {
            "accepted": max(errors.values()) < 1e-8,
            "reason": "original equation gates",
            "dtype": str(np.dtype(dtype)),
            "linear_solves": iteration + 1,
            "physical_unknowns": nv + nj,
            "normal_rows": count,
            "active_rows": len(ids),
            "errors": errors,
            "steps": stats,
            "velocity": vd.tolist(),
            "joint": jd.tolist(),
            "normal": nd.tolist(),
        }
    return {"accepted": False, "reason": "bounded active-set budget exhausted", "steps": stats}


def main():
    z = np.load("/tmp/colibri_colored_block_dispatch_trace600.phases.npz")
    phase = "biased"
    d = {k.split(".", 1)[1]: z[k] for k in z.files if k.startswith(phase + "_after.")}
    d["copy_section_end"] = np.arange(len(d["velocity"]))
    d["copy_velocity"] = d["velocity"][1:]
    d["copy_angular_velocity"] = d["angular_velocity"][1:]
    a = assemble_snapshot(d, phase, float(z["dt"][0]), int(z["num_joints"][0]))
    result = {
        "scope": "Frozen original joint+normal subproblem, tangential impulses held; not full Coulomb acceptance",
        "fp64_oracle": solve_normal(a, np.float64),
        "fp32_candidate": solve_normal(a, np.float32),
        "fp32_whitened": solve_normal(a, np.float32, whiten=True),
    }
    Path("/tmp/colibri_colored_joint_normal_block.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
