# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""FP32 C-then-J iteration on the accepted current colored phase snapshots."""

import json
from pathlib import Path

import numpy as np
import warp as wp

from .check_colored_d6_fp32 import original_residuals, sweep
from .coupled_support_online import assemble_snapshot


def main():
    path = Path("/tmp/colibri_colored_block_dispatch_trace600.phases.npz")
    z = np.load(path)
    result = {
        "source": str(path),
        "scope": "Frozen current equations, original C then J order, FP32 numerical iterates; no live claim",
        "cases": [],
    }
    for phase in ("biased", "relax"):
        label = phase + "_after"
        d = {k.split(".", 1)[1]: z[k] for k in z.files if k.startswith(label + ".")}
        d["copy_section_end"] = np.arange(len(d["velocity"]), dtype=np.int32)
        d["copy_velocity"] = d["velocity"][1:]
        d["copy_angular_velocity"] = d["angular_velocity"][1:]
        a = assemble_snapshot(d, phase, float(z["dt"][0]), int(z["num_joints"][0]))
        w, b, c, r, target = [a[k].astype(np.float32) for k in ("W", "B", "C", "diagonal", "targets")]
        bias = (a["rhs"] - a["C"] @ a["vbar"]).astype(np.float32)
        wc = w @ c.T
        h = c @ wc
        k = b @ w @ b.T + np.diag(r)
        v = a["velocity"].astype(np.float32)
        joint = a["old_joint"].astype(np.float32)
        lam = a["old"].astype(np.float32)
        vstart = v.copy()
        jstart = joint.copy()
        lstart = lam.copy()
        arrays = [
            wp.array(x, dtype=wp.float32, device="cpu")
            for x in (c, wc, h, bias, a["gamma"].astype(np.float32), a["mu"].astype(np.float32))
        ]
        records = [dict(iterations=0, **original_residuals(a, v, joint, lam, bias))]
        for iteration in range(1, 65):
            va = wp.array(v, dtype=wp.float32, device="cpu")
            la = wp.array(lam, dtype=wp.float32, device="cpu")
            wp.launch(sweep, 1, inputs=[*arrays, la, va, len(lam) // 3, len(v)], device="cpu")
            v, lam = va.numpy(), la.numpy()
            delta = np.linalg.solve(k, -(b @ v + r * joint - target))
            joint += delta
            v += w @ b.T @ delta
            if iteration in (1, 2, 4, 8, 16, 32, 64):
                ledger = np.max(
                    abs(
                        v.astype(float)
                        - vstart
                        - a["W"] @ (a["B"].T @ (joint.astype(float) - jstart) + a["C"].T @ (lam.astype(float) - lstart))
                    )
                )
                assert ledger < 2e-6, ledger
                records.append(
                    dict(
                        iterations=iteration,
                        impulse_response_error=float(ledger),
                        base_velocity=v[:3].astype(float).tolist(),
                        **original_residuals(a, v, joint, lam, bias),
                    )
                )
        result["cases"].append({"phase": phase, "contacts": len(lam) // 3, "records": records})
    Path("/tmp/colibri_current_colored_sweeps.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
