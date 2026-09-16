# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Read-only comparable final-snapshot point-load and friction-history metrics."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def quantiles(x):
    return (
        dict(zip(("min", "p25", "median", "p75", "max"), np.quantile(x, [0, 0.25, 0.5, 0.75, 1]).tolist(), strict=True))
        if len(x)
        else None
    )


def audit(prefix):
    d = np.load(prefix + ".native_state.npz")
    meta = json.loads(Path(prefix + ".native_conditioned.json").read_text())
    trajectory = np.load(prefix + ".npz")
    labels = ["static", *trajectory["labels"].tolist()]
    count = int(d["contact_valid_count"][0])
    headers = d["contact_columns_data"].view(np.int32)
    rows = []
    covered = set()
    for column in range(headers.shape[1]):
        if headers[0, column] != 9:
            continue
        b0, b1 = map(int, headers[1:3, column])
        first, n = map(int, headers[5:7, column])
        mu, mu_dynamic = map(float, d["contact_columns_data"][3:5, column])
        for k in range(first, first + n):
            assert k < count and k not in covered
            covered.add(k)
            x = d["contact_lambdas"][:, k].astype(float)
            lam = d["contact_impulses"][:, k].astype(float)
            normal, t0 = x[:3], x[3:6]
            t1 = np.cross(normal, t0)
            tangent_impulse = lam[1] * t0 + lam[2] * t1
            normal_impulse = lam[0] * normal
            dynamic_body = b0 if b1 == 0 else b1 if b0 == 0 else -1
            sign = -1 if dynamic_body == b0 else 1
            lever_start = 9 if dynamic_body == b0 else 12
            r = d["contact_derived"][lever_start : lever_start + 3, k].astype(float)
            position = d["body_position"][dynamic_body].astype(float) if dynamic_body >= 0 else np.zeros(3)
            capacity = mu * lam[0]
            ratio = float(np.linalg.norm(lam[1:]) / capacity) if capacity > 0 else None
            reference_motion = []
            for body, ps in ((b0, 13), (b1, 16)):
                reference_motion.append(float(np.linalg.norm(d["body_position"][body].astype(float) - x[ps : ps + 3])))
            rows.append(
                {
                    "point": k,
                    "column": column,
                    "bodies": [b0, b1],
                    "body_ground": dynamic_body,
                    "mu_static": mu,
                    "mu_dynamic": mu_dynamic,
                    "normal_impulse_Ns": float(lam[0]),
                    "tangent_impulse_norm_Ns": float(np.linalg.norm(lam[1:])),
                    "capacity_Ns": capacity,
                    "capacity_fraction": ratio,
                    "broken": bool(x[12]),
                    "match_index": int(d["contact_views_rigid_contact_match_index"][k]),
                    "normal_impulse_world_on_body_Ns": (sign * normal_impulse).tolist(),
                    "tangent_impulse_world_on_body_Ns": (sign * tangent_impulse).tolist(),
                    "normal_torque_impulse_about_COM_Nms": np.cross(r, sign * normal_impulse).tolist(),
                    "tangent_torque_impulse_about_COM_Nms": np.cross(r, sign * tangent_impulse).tolist(),
                    "normal_torque_impulse_about_origin_Nms": np.cross(position + r, sign * normal_impulse).tolist(),
                    "birth_to_current_COM_distance_m": reference_motion,
                    "birth_pose_signature": hashlib.sha256(d["contact_lambdas"][13:27, k].tobytes()).hexdigest(),
                    "stored_start_gap_m": float(d["contact_derived"][15, k]),
                }
            )
    assert len(covered) == count
    summaries = []
    for body in sorted({r["body_ground"] for r in rows}):
        for material in ("all", "positive_mu", "zero_mu"):
            group = [
                r
                for r in rows
                if r["body_ground"] == body
                and (material == "all" or (r["mu_static"] > 0) == (material == "positive_mu"))
            ]
            if not group:
                continue
            loaded = [r for r in group if r["normal_impulse_Ns"] > 0]
            loads = np.array([r["normal_impulse_Ns"] for r in group])
            total = loads.sum()
            caps = sum(r["capacity_Ns"] for r in group)
            tangent = np.sum([r["tangent_impulse_world_on_body_Ns"] for r in group], axis=0)
            sum_tangent = sum(r["tangent_impulse_norm_Ns"] for r in group)
            broken = [r for r in group if r["broken"]]
            s = {
                "body": body,
                "label": labels[body] if body >= 0 else "two_dynamic",
                "material_subset": material,
                "points": len(group),
                "loaded_points": len(loaded),
                "zero_normal_points": len(group) - len(loaded),
                "normal_impulse_total_Ns": float(total),
                "equivalent_force_at_h_1_3600_N": float(total * 3600),
                "loaded_normal_impulse_quantiles_Ns": quantiles([r["normal_impulse_Ns"] for r in loaded]),
                "effective_loaded_point_count": float(total**2 / np.dot(loads, loads)) if total else 0,
                "top1_normal_fraction": float(max(loads) / total) if total else 0,
                "top4_normal_fraction": float(np.sort(loads)[-4:].sum() / total) if total else 0,
                "broken_points": len(broken),
                "broken_loaded_points": sum(r["normal_impulse_Ns"] > 0 for r in broken),
                "broken_unloaded_points": sum(r["normal_impulse_Ns"] == 0 for r in broken),
                "broken_normal_fraction": sum(r["normal_impulse_Ns"] for r in broken) / total if total else 0,
                "loaded_near_98pct_cap": sum(
                    r["capacity_fraction"] is not None and r["capacity_fraction"] >= 0.98 for r in loaded
                ),
                "loaded_capacity_fraction_quantiles": quantiles(
                    [r["capacity_fraction"] for r in loaded if r["capacity_fraction"] is not None]
                ),
                "local_capacity_total_Ns": caps,
                "sum_tangent_norm_over_total_capacity": sum_tangent / caps if caps else None,
                "net_tangent_norm_over_total_capacity": float(np.linalg.norm(tangent) / caps) if caps else None,
                "birth_pose_signature_count": len({r["birth_pose_signature"] for r in group}),
                "loaded_birth_pose_signature_count": len({r["birth_pose_signature"] for r in loaded}),
                "match_index_counts": {str(k): sum(r["match_index"] == k for r in group) for k in (-2, -1)},
                "nonnegative_match_indices": sum(r["match_index"] >= 0 for r in group),
            }
            for key in (
                "normal_impulse_world_on_body_Ns",
                "tangent_impulse_world_on_body_Ns",
                "normal_torque_impulse_about_COM_Nms",
                "tangent_torque_impulse_about_COM_Nms",
                "normal_torque_impulse_about_origin_Nms",
            ):
                s[key] = np.sum([r[key] for r in group], axis=0).tolist()
            summaries.append(s)
    return {
        "source": prefix,
        "phase": "Saved final solver state after final relaxation; impulses are accumulated row values, not a time-integrated load average.",
        "scope": "Final-state distribution only; no causal creep, patch equivalence, full static feasibility or history-duration claim.",
        "match_index_limitation": "Raw collision match index is sticky geometry state, not the separate history-only identity map; negative values alone do not demonstrate history loss.",
        "history_limitation": "14-float reference poses contain no birth timestamp. Distinct poses/signatures and displacement are observable; elapsed lifetimes cannot be recovered reliably from a stationary/revisiting trajectory.",
        "wrench_convention": "Impulse on body0 = -(lambda_n*n+lambda_t1*t1+lambda_t2*t2); both use saved common-forcepoint levers. Normal force lambda/h is a labeled per-substep equivalent only.",
        "actual_owned_binding_gate": meta.get("owned_binding_gate"),
        "production_unchanged": meta.get("production_unchanged"),
        "summaries": summaries,
        "rows": rows,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prefix", default="/tmp/colibri_native_direct_owned_fixed3600")
    parser.add_argument("--output", default="/tmp/colibri_corrected_final_contact_distribution.json")
    args = parser.parse_args()
    report = audit(args.prefix)
    Path(args.output).write_text(json.dumps(report, indent=2))
    print(json.dumps(report["summaries"], indent=2))


if __name__ == "__main__":
    main()
