# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""CPU FP32 full-coordinate joint/contact coupling gate on original frozen rows.

Uses the native implicit-drive equation and metric point projection at SOR1.
Offline double arithmetic checks residual/impulse conservation only.
"""

import json
from pathlib import Path

import numpy as np
import warp as wp

from newton._src.solvers.phoenx.constraints.contact_projection import contact_project_friction_metric

from .coupled_support_online import assemble_snapshot


@wp.kernel(enable_backward=False)
def sweep(
    c: wp.array2d[wp.float32],
    wc: wp.array2d[wp.float32],
    h: wp.array2d[wp.float32],
    bias: wp.array[wp.float32],
    gamma: wp.array[wp.float32],
    mu: wp.array[wp.float32],
    lam: wp.array[wp.float32],
    v: wp.array[wp.float32],
    points: int,
    size: int,
):
    for p in range(points):
        row = 3 * p
        vn = bias[row]
        for j in range(size):
            vn += c[row, j] * v[j]
        old = lam[row]
        new = wp.max(wp.float32(0), old - (vn + gamma[p] * old) / (h[row, row] + gamma[p]))
        lam[row] = new
        for j in range(size):
            v[j] += wc[j, row] * (new - old)
        vt0 = bias[row + 1]
        vt1 = bias[row + 2]
        for j in range(size):
            vt0 += c[row + 1, j] * v[j]
            vt1 += c[row + 2, j] * v[j]
        old_t0 = lam[row + 1]
        old_t1 = lam[row + 2]
        t = contact_project_friction_metric(
            h[row + 1, row + 1],
            h[row + 1, row + 2],
            h[row + 2, row + 2],
            vt0,
            vt1,
            old_t0,
            old_t1,
            mu[p] * new,
            mu[p] * new,
        )
        lam[row + 1] = t[0]
        lam[row + 2] = t[1]
        for j in range(size):
            v[j] += wc[j, row + 1] * (t[0] - old_t0) + wc[j, row + 2] * (t[1] - old_t1)


def original_residuals(a, v, joint, lam, bias):
    j = a["B"] @ v.astype(float) + a["diagonal"] * joint - a["targets"]
    g = (a["C"] @ v.astype(float) + bias).reshape(-1, 3)
    g[:, 0] += a["gamma"] * lam[::3]
    triples = lam.reshape(-1, 3).astype(float)
    # Original non-associated natural-map condition with scalar positive scale.
    scale = np.max(np.diag(a["C"] @ a["W"] @ a["C"].T).reshape(-1, 3), axis=1) + a["gamma"]
    normal = scale * (triples[:, 0] - np.maximum(0, triples[:, 0] - g[:, 0] / scale))
    trial = triples[:, 1:] - g[:, 1:] / scale[:, None]
    length = np.linalg.norm(trial, axis=1)
    cap = a["mu"] * np.maximum(triples[:, 0], 0)
    projection = trial * np.minimum(1, np.divide(cap, length, out=np.ones_like(cap), where=length > 0))[:, None]
    tangent = scale[:, None] * (triples[:, 1:] - projection)
    return {
        "joint_rows": j.tolist(),
        "hard_joint_max": float(np.max(abs(j[a["diagonal"] == 0]))),
        "drive_max": float(np.max(abs(j[a["diagonal"] > 0]))),
        "contact_natural_max": float(max(np.max(abs(normal)), np.max(abs(tangent)))),
    }


def main():
    source = Path("/tmp/colibri_base_frame_totalnormal_phases330.npz")
    z = np.load(source)
    result = {
        "source": str(source),
        "scope": "Historical frozen67/29point equations, not corrected currenttrajectory or GPUtiming. FP32 candidate, FP64 independent residual oracle.",
        "cases": [],
    }
    for phase in ("biased", "relax"):
        d = {k.split(".", 1)[1]: z[k] for k in z.files if k.startswith(phase + "_solved.")}
        a = assemble_snapshot(d, phase, float(z["dt"][0]), int(z["num_joints"][0]))
        # All candidate operator inputs/updates remain FP32.
        w, b, c, r, target = [np.asarray(a[k], np.float32) for k in ("W", "B", "C", "diagonal", "targets")]
        gamma = np.asarray(a["gamma"], np.float32)
        v = np.asarray(a["velocity"], np.float32)
        joint = np.asarray(a["old_joint"], np.float32)
        lam = np.asarray(a["old"], np.float32)
        start = v.copy()
        jstart = joint.copy()
        lstart = lam.copy()
        k = b @ w @ b.T + np.diag(r)
        wc = w @ c.T
        h = c @ wc
        bias = np.asarray(a["rhs"] - a["C"] @ a["vbar"], np.float32)
        arrays = [
            wp.array(x, dtype=wp.float32, device="cpu")
            for x in (c, wc, h, bias, gamma, np.asarray(a["mu"], np.float32))
        ]
        records = []
        for iteration in range(1, 33):
            residual = b @ v + r * joint - target
            delta = np.linalg.solve(k, -residual)
            assert delta.dtype == np.float32
            joint += delta
            v += w @ b.T @ delta
            before = original_residuals(a, v, joint, lam, bias)
            va = wp.array(v, dtype=wp.float32, device="cpu")
            la = wp.array(lam, dtype=wp.float32, device="cpu")
            wp.launch(sweep, 1, inputs=[*arrays, la, va, len(lam) // 3, len(v)], device="cpu")
            v, lam = va.numpy(), la.numpy()
            if iteration in (1, 2, 4, 8, 16, 32):
                after = original_residuals(a, v, joint, lam, bias)
                impulse = a["B"].T @ (joint.astype(float) - jstart) + a["C"].T @ (lam.astype(float) - lstart)
                physical_error = float(np.max(abs(v.astype(float) - start - a["W"] @ impulse)))
                assert physical_error < 2.0e-6, physical_error  # FP32 accumulated-scatter check.
                records.append(
                    {
                        "sweep": iteration,
                        "immediately_after_joint": before,
                        "after_contacts": after,
                        "impulse_response_velocity_error": physical_error,
                    }
                )
        result["cases"].append({"phase": phase, "contacts": len(lam) // 3, "records": records})
    Path("/tmp/colibri_colored_d6_fp32.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
