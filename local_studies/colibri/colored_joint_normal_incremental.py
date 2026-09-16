# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""FP32 incremental joint/normal solve with an independent applied-impulse audit."""

import json
from pathlib import Path

import numpy as np

from .colored_joint_normal_block import solve_equilibrated, solve_normal
from .coupled_support_online import assemble_snapshot


def compensated_matvec(matrix, vector):
    """Sum FP32 row products with compensated accumulation."""
    total = np.zeros(matrix.shape[0], dtype=np.float32)
    correction = total.copy()
    for k in range(matrix.shape[1]):
        left, right = matrix[:, k], vector[k]
        value = left * right
        left_split = np.float32(4097) * left
        left_hi = left_split - (left_split - left)
        left_lo = left - left_hi
        right_split = np.float32(4097) * right
        right_hi = right_split - (right_split - right)
        right_lo = right - right_hi
        product_error = ((left_hi * right_hi - value) + left_hi * right_lo + left_lo * right_hi) + left_lo * right_lo
        correction += product_error
        updated = total + value
        correction += np.where(abs(total) >= abs(value), (total - updated) + value, (value - updated) + total)
        total = updated
    return total + correction


def solve_incremental(a, refinements=3, compensate=True):
    """Refine original equations through paired impulse scatter, never overwrite velocity."""
    f = np.float32
    w, b, c = [a[k].astype(f) for k in ("W", "B", "C")]
    c = c[::3]
    gamma = a["gamma"].astype(f)
    bias = (a["rhs"] - a["C"] @ a["vbar"])[::3].astype(f)
    diagonal, target = [a[k].astype(f) for k in ("diagonal", "targets")]
    v = a["velocity"].astype(f).copy()
    j = a["old_joint"].astype(f).copy()
    n = a["old"][::3].astype(f).copy()
    seed = solve_normal(a, np.float32)
    if "normal" not in seed:
        return {"accepted": False, "reason": seed["reason"], "records": []}
    active = np.array(seed["normal"]) > 0
    mass = np.linalg.inv(w)
    nv, nj = len(v), len(j)
    impulse_ledger = np.zeros(nv, dtype=np.float64)  # Audit only; never feeds solve.
    records = []
    for _iteration in range(refinements):
        ids = np.flatnonzero(active)
        inactive = np.flatnonzero(~active)
        gn = c @ v + bias + gamma * n
        gj = b @ v + diagonal * j - target
        scaled = c[ids] / gamma[ids, None]
        h = mass + c[ids].T @ scaled
        matrix = np.zeros((nv + nj, nv + nj), dtype=f)
        matrix[:nv, :nv] = h
        matrix[:nv, nv:] = -b.T
        matrix[nv:, :nv] = b
        matrix[nv:, nv:] = np.diag(diagonal)
        rhs = -(scaled.T @ gn[ids]) - c[inactive].T @ n[inactive]
        solution = solve_equilibrated(matrix, np.r_[rhs, -gj])
        dj = solution[nv:]
        dn = -n.copy()
        dn[ids] = -(gn[ids] + c[ids] @ solution[:nv]) / gamma[ids]
        # The applied increment is distinct from rounded accumulated multipliers.
        impulse = (
            compensated_matvec(np.concatenate((b.T, c.T), axis=1), np.r_[dj, dn]) if compensate else b.T @ dj + c.T @ dn
        )
        delta = w @ impulse
        v += delta
        j += dj
        n += dn
        impulse_ledger += b.astype(float).T @ dj.astype(float) + c.astype(float).T @ dn.astype(float)
        bad = n < 0
        active[bad] = False
        gradient = c @ v + bias + gamma * n
        active[(~active) & (gradient < -1e-8)] = True
        vd = v.astype(float)
        gd = a["C"][::3] @ vd + (a["rhs"] - a["C"] @ a["vbar"])[::3] + a["gamma"] * n.astype(float)
        natural = np.where(n > 0, gd, np.minimum(gd, 0))
        errors = {
            "normal": float(np.max(abs(natural))),
            "joint": float(np.max(abs(a["B"] @ vd + a["diagonal"] * j - a["targets"]))),
            "applied_scatter": float(np.max(abs(vd - a["velocity"] - a["W"] @ impulse_ledger))),
            "negative": float(max(0, -min(n))),
        }
        mass64 = np.linalg.inv(a["W"])
        actual_delta = vd - a["velocity"]
        wrench_error = mass64 @ actual_delta - impulse_ledger
        errors["linear_momentum_error"] = float(np.max(abs(wrench_error.reshape(-1, 6)[:, :3].sum(axis=0))))
        errors["body_angular_momentum_error"] = float(np.max(abs(wrench_error.reshape(-1, 6)[:, 3:])))
        midpoint = (vd + a["velocity"]) / 2
        errors["work_error"] = float(abs(actual_delta @ mass64 @ midpoint - impulse_ledger @ midpoint))
        records.append(errors)
    return {
        "accepted": max(records[-1].values()) < 1e-8,
        "records": records,
        "velocity": v.tolist(),
        "joint": j.tolist(),
        "normal": n.tolist(),
    }


def main():
    """Compare original absolute solve and incremental scatter on both captured phases."""
    z = np.load("/tmp/colibri_colored_block_dispatch_trace600.phases.npz")
    results = {}
    for phase in ("biased",):
        d = {k.split(".", 1)[1]: z[k] for k in z.files if k.startswith(phase + "_after.")}
        d["copy_section_end"] = np.arange(len(d["velocity"]))
        d["copy_velocity"] = d["velocity"][1:]
        d["copy_angular_velocity"] = d["angular_velocity"][1:]
        a = assemble_snapshot(d, phase, float(z["dt"][0]), int(z["num_joints"][0]))
        results[phase] = solve_incremental(a)
    Path("/tmp/colibri_colored_joint_normal_incremental.json").write_text(json.dumps(results, indent=2))
    print(json.dumps({p: r["records"] for p, r in results.items()}, indent=2))


if __name__ == "__main__":
    main()
