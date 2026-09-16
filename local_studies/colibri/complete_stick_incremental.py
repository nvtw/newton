# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Bounded FP32 full-stick candidate with every original point cone retained.

Unbiased hard-normal support only. The supported body is discovered from the
contact Jacobian, never a scene name. Unknown feasibility falls back unchanged.
FP64 calculations are independent acceptance audits, never solver inputs.
"""

import json
from pathlib import Path

import numpy as np

from .colored_joint_normal_block import solve_equilibrated
from .colored_joint_normal_incremental import compensated_matvec
from .coupled_support_online import assemble_snapshot


def propose(a, refinements=3, compensate=True, uniform_weights=False):
    """Solve physical body/joint/support increments and redistribute actual point impulses."""
    f = np.float32
    w, b, c = [a[k].astype(f) for k in ("W", "B", "C")]
    nv, nj = len(w), len(b)
    nonzero = np.flatnonzero(np.any(c != 0, axis=0))
    support_bodies = np.unique(nonzero // 6)
    if len(support_bodies) != 1 or np.any(a["gamma"] != 0):
        return {"accepted": False, "reason": "Requires one hard-normal support body", "records": []}
    body = int(support_bodies[0])
    support = np.zeros((6, nv), dtype=f)
    support[:, 6 * body : 6 * body + 6] = np.eye(6, dtype=f)
    point_map = c[:, 6 * body : 6 * body + 6].T
    contact_bias = (a["rhs"] - a["C"] @ a["vbar"]).astype(f)
    if np.max(abs(contact_bias)) > 1e-12:
        return {"accepted": False, "reason": "Nonzero contact targets require compatible affine solve", "records": []}
    diagonal, target = [a[k].astype(f) for k in ("diagonal", "targets")]
    mass = np.linalg.inv(w)
    matrix = np.zeros((nv + nj + 6, nv + nj + 6), dtype=f)
    matrix[:nv, :nv] = mass
    matrix[:nv, nv : nv + nj] = -b.T
    matrix[:nv, nv + nj :] = -support.T
    matrix[nv : nv + nj, :nv] = b
    matrix[nv : nv + nj, nv : nv + nj] = np.diag(diagonal)
    matrix[nv + nj :, :nv] = support
    v = a["velocity"].astype(f).copy()
    joint = a["old_joint"].astype(f).copy()
    impulse = a["old"].astype(f).copy()
    weights = np.repeat(np.maximum(impulse[::3], f(0)), 3)
    if np.max(weights) <= 0:
        return {"accepted": False, "reason": "No positive supported normal load", "records": []}
    weights /= np.max(weights)
    if uniform_weights:
        weights.fill(f(1))
    gram = (point_map * weights) @ point_map.T
    try:
        factor = np.linalg.cholesky(gram)
    except np.linalg.LinAlgError:
        return {"accepted": False, "reason": "Insufficient original point wrench span", "records": []}
    ledger = np.zeros(nv, dtype=np.float64)
    records = []
    for _iteration in range(refinements):
        rhs = np.r_[np.zeros(nv, dtype=f), -(b @ v + diagonal * joint - target), -support @ v]
        solution = solve_equilibrated(matrix, rhs)
        dj = solution[nv : nv + nj]
        dw = solution[nv + nj :]
        target_wrench = compensated_matvec(point_map, impulse) + dw
        dual = np.linalg.solve(factor.T, np.linalg.solve(factor, target_wrench))
        proposed = weights * (point_map.T @ dual)
        dl = proposed - impulse
        jac = np.c_[b.T, c.T]
        increments = np.r_[dj, dl]
        applied = compensated_matvec(jac, increments) if compensate else jac @ increments
        v += w @ applied
        joint += dj
        impulse += dl
        ledger += b.astype(float).T @ dj.astype(float) + c.astype(float).T @ dl.astype(float)
        vd = v.astype(float)
        lam = impulse.astype(float).reshape(-1, 3)
        contact_v = a["C"] @ vd + contact_bias.astype(float)
        cone = np.linalg.norm(lam[:, 1:], axis=1) - a["mu"] * lam[:, 0]
        normal = contact_v[::3]
        natural = np.where(lam[:, 0] > 0, normal, np.minimum(normal, 0))
        residual = vd - a["velocity"] - a["W"] @ ledger
        delta = vd - a["velocity"]
        md = np.linalg.inv(a["W"])
        wrench_error = md @ delta - ledger
        errors = {
            "normal": float(np.max(abs(natural))),
            "tangent": float(np.max(abs(contact_v.reshape(-1, 3)[:, 1:]))),
            "joint": float(np.max(abs(a["B"] @ vd + a["diagonal"] * joint - a["targets"]))),
            "scatter": float(np.max(abs(residual))),
            "negative": float(max(0, -np.min(lam[:, 0]))),
            "cone_excess": float(max(0, np.max(cone))),
            "linear_momentum": float(np.max(abs(wrench_error.reshape(-1, 6)[:, :3].sum(axis=0)))),
            "body_angular_momentum": float(np.max(abs(wrench_error.reshape(-1, 6)[:, 3:]))),
            "work": float(abs((md @ delta - ledger) @ ((vd + a["velocity"]) / 2))),
        }
        records.append(errors)
    accepted = (
        max(records[-1].values()) < 1e-8 and records[-1]["cone_excess"] <= 1e-12 and records[-1]["negative"] <= 1e-12
    )
    return {
        "accepted": accepted,
        "reason": "Original complete sticking gates" if accepted else "UNKNOWN: candidate failed original gates",
        "records": records,
        "velocity": v.tolist(),
        "joint": joint.tolist(),
        "impulse": impulse.tolist(),
        "unknowns": nv + nj + 6,
        "points": len(impulse) // 3,
        "scope": "Eligible unbiased points; skipped speculative rows audited separately",
    }


def frozen():
    """Load the exact captured unbiased point/joint equations."""
    z = np.load("/tmp/colibri_colored_block_dispatch_trace600.phases.npz")
    d = {k.split(".", 1)[1]: z[k] for k in z.files if k.startswith("relax_after.")}
    d["copy_section_end"] = np.arange(len(d["velocity"]))
    d["copy_velocity"] = d["velocity"][1:]
    d["copy_angular_velocity"] = d["angular_velocity"][1:]
    return assemble_snapshot(d, "relax", float(z["dt"][0]), int(z["num_joints"][0]))


def main():
    """Run a complete sticking candidate on the frozen native equations."""
    a = frozen()
    result = propose(a)
    Path("/tmp/colibri_complete_stick_incremental.json").write_text(json.dumps(result, indent=2))
    print(json.dumps({k: v for k, v in result.items() if k not in ("velocity", "joint", "impulse")}, indent=2))


if __name__ == "__main__":
    main()
