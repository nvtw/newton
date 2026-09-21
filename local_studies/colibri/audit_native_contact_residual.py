"""Read final actual contact residuals, separating held speculative rows."""

import argparse
import json
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("snapshot", type=Path)
    args = parser.parse_args()
    x = np.load(args.snapshot)
    headers = x["contact_columns_data"]
    h = headers.view(np.int32)
    count = int(x["contact_valid_count"][0])
    lam = x["contact_impulses"].astype(float)
    data = x["contact_lambdas"].astype(float)
    derived = x["contact_derived"].astype(float)
    v = x["body_velocity"].astype(float)
    w = x["body_angular_velocity"].astype(float)
    rows = []
    seen = set()
    for col in range(h.shape[1]):
        first, n = h[5:7, col]
        if n <= 0 or first < 0 or first + n > count:
            continue
        i, j = h[1:3, col]
        for k in range(first, first + n):
            if k in seen:
                continue
            seen.add(k)
            normal = data[:3, k]
            t1 = data[3:6, k]
            t2 = np.cross(normal, t1)
            relative = v[j] + np.cross(w[j], derived[12:15, k]) - v[i] - np.cross(w[i], derived[9:12, k])
            vt = np.array([relative @ t1, relative @ t2])
            load = lam[0, k]
            tangent = lam[1:3, k]
            radius = float(headers[3, col]) * load
            length = float(np.linalg.norm(tangent))
            bound = 16 * np.finfo(np.float32).eps * max(abs(radius), length, np.finfo(np.float32).tiny)
            rows.append(
                {
                    "point": int(k),
                    "body_slots": [int(i), int(j)],
                    "normal_impulse_Ns": float(load),
                    "tangent_impulse_Ns": tangent.tolist(),
                    "static_radius_Ns": radius,
                    "cone_fraction": length / radius if radius > 0 else None,
                    "broken": bool(data[12, k]),
                    "tangent_bias_m_s": derived[4:6, k].tolist(),
                    "biased_tangent_gradient_m_s": (vt + derived[4:6, k]).tolist(),
                    "interior_beyond_roundoff": bool(length < radius - bound),
                    "cone_excess_Ns": max(0.0, length - radius),
                    "vn_m_s": float(relative @ normal),
                    "vt_m_s": vt.tolist(),
                    "vt_norm_m_s": float(np.linalg.norm(vt)),
                    "bias_m_s": float(derived[3, k]),
                    "held_speculative_in_relax": bool(derived[3, k] > 0),
                }
            )
    overlap = [r for r in rows if not r["held_speculative_in_relax"] and r["normal_impulse_Ns"] > 0]
    interior = [r for r in overlap if r["interior_beyond_roundoff"]]
    report = {
        "snapshot": str(args.snapshot),
        "scope": "Final current body velocities and prepared common-point levers; overlap rows evaluated against unbiased final-relax law. Speculative rows reported separately because native relax holds them fixed. Not an impulse-work ledger.",
        "count": count,
        "covered": len(seen),
        "positive_loaded_overlap_count": len(overlap),
        "interior_loaded_overlap_count": len(interior),
        "max_interior_loaded_overlap_slip_m_s": max((r["vt_norm_m_s"] for r in interior), default=0.0),
        "max_loaded_overlap_abs_vn_m_s": max((abs(r["vn_m_s"]) for r in overlap), default=0.0),
        "max_cone_excess_Ns": max((r["cone_excess_Ns"] for r in rows), default=0.0),
        "broken_count": sum(r["broken"] for r in rows),
        "max_tangent_bias_m_s": float(np.max(np.abs(derived[4:6, :count]))),
        "per_body": {
            str(body): {
                "contacts": sum(body in r["body_slots"] for r in rows),
                "normal_impulse_Ns": sum(r["normal_impulse_Ns"] for r in rows if body in r["body_slots"]),
                "broken": sum(r["broken"] for r in rows if body in r["body_slots"]),
                "loaded_overlap": sum(body in r["body_slots"] for r in overlap),
                "max_interior_overlap_slip_m_s": max(
                    (r["vt_norm_m_s"] for r in interior if body in r["body_slots"]), default=0.0
                ),
            }
            for body in sorted({b for r in rows for b in r["body_slots"] if b != 0})
        },
        "rows": rows,
    }
    assert len(seen) == count
    args.snapshot.with_suffix(".contact_residual.json").write_text(json.dumps(report, indent=2))
    print({k: v for k, v in report.items() if k != "rows"})


if __name__ == "__main__":
    main()
