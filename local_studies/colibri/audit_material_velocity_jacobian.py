# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Compare a saved material-reference derivative with native impulse-point rows."""

import json
from pathlib import Path

import numpy as np

from .check_friction_differential_reference import conjugate, multiply, normalize, rotate


def main():
    """Differentiate the exact normalized material function without a native run."""
    prefix = "/tmp/colibri_native_direct_normalized3600"
    d = np.load(prefix + ".native_state.npz")
    meta = json.load(open(prefix + ".native_conditioned.json"))
    com = np.vstack([np.zeros(3), np.array(meta["properties"]["body_com"])])
    n = int(d["contact_views_rigid_contact_count"][0])
    side0, side1 = np.full(n, -1), np.full(n, -1)
    headers = d["contact_columns_data"].view(np.int32)
    for col in range(headers.shape[1]):
        if headers[0, col] != 9:
            continue
        start, count = headers[5:7, col]
        side0[start : start + count] = headers[1, col]
        side1[start : start + count] = headers[2, col]
    assert np.all(side0 >= 0) and np.all(side1 >= 0)
    rows = []
    for k, (b0, b1) in enumerate(zip(side0, side1, strict=True)):
        x = d["contact_lambdas"][:, k].astype(float)
        p0, p1 = d["body_position"][[b0, b1]].astype(float)
        q0, q1 = [normalize(v.astype(float)) for v in d["body_orientation"][[b0, b1]]]
        pb0, pb1 = x[13:16], x[16:19]
        qb0, qb1 = normalize(x[19:23]), normalize(x[23:27])
        a1 = x[9:12] - com[b1]
        # Exact normalized-reference effective body0 anchor; stored rounded
        # anchor0 is deliberately not used by the differential candidate.
        a0 = rotate(conjugate(qb0), pb1 - pb0) + rotate(multiply(conjugate(qb0), qb1), a1)
        r0, r1 = rotate(q0, a0), rotate(q1, a1)
        point0, point1 = p0 + r0, p1 + r1
        drift = point1 - point0
        native0 = d["contact_derived"][9:12, k].astype(float)
        native1 = d["contact_derived"][12:15, k].astype(float)
        v0, v1 = d["body_velocity"][[b0, b1]].astype(float)
        w0, w1 = d["body_angular_velocity"][[b0, b1]].astype(float)
        material_rate = v1 + np.cross(w1, r1) - v0 - np.cross(w0, r0)
        native_rate = v1 + np.cross(w1, native1) - v0 - np.cross(w0, native0)
        t = np.array([x[3:6], np.cross(x[:3], x[3:6])])
        midpoint = 0.5 * (point0 + point1)
        midpoint_rate = v1 + np.cross(w1, midpoint - p1) - v0 - np.cross(w0, midpoint - p0)
        rows.append(
            {
                "point": k,
                "bodies": [int(b0), int(b1)],
                "loaded": bool(d["contact_impulses"][0, k] > 0),
                "normal_impulse": float(d["contact_impulses"][0, k]),
                "material_world_drift_m": drift.tolist(),
                "material_tangent_drift_m": (t @ drift).tolist(),
                "material_lever_minus_native0_m": (r0 - native0).tolist(),
                "material_lever_minus_native1_m": (r1 - native1).tolist(),
                "material_rate_mps": (t @ material_rate).tolist(),
                "native_rate_mps": (t @ native_rate).tolist(),
                "rate_difference_mps": (t @ (material_rate - native_rate)).tolist(),
                "common_material_midpoint_rate_difference_mps": (t @ (material_rate - midpoint_rate)).tolist(),
                "implied_friction_bias_mps": (288 * t @ drift).tolist(),
                "native_stored_friction_bias_mps": d["contact_derived"][4:6, k].tolist(),
                "source_gap_bias": float(d["contact_derived"][3, k]),
            }
        )
    summary = []
    for body in np.unique(side0):
        group = [r for r in rows if r["bodies"][0] == body and r["loaded"]]
        if not group:
            continue
        keys = [
            "material_tangent_drift_m",
            "material_lever_minus_native0_m",
            "material_lever_minus_native1_m",
            "rate_difference_mps",
            "common_material_midpoint_rate_difference_mps",
            "implied_friction_bias_mps",
        ]
        summary.append(
            {
                "body": int(body),
                "loaded_points": len(group),
                "maxima": {key: float(np.max(np.abs([r[key] for r in group]))) for key in keys},
            }
        )
    report = {
        "scope": "Partial candidate: owned callback installed late. Offline final-pose geometry only; not combined-history or causal creep evidence.",
        "source": prefix,
        "summary": summary,
        "rows": rows,
    }
    Path("/tmp/colibri_material_velocity_jacobian.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
