"""Offline physical P/L and midpoint-work audit of the hybrid friction phase."""

import argparse
import json
from pathlib import Path

import numpy as np


def rotation(q):
    q = q / np.linalg.norm(q)
    x, y, z, w = q
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("snapshot", type=Path)
    a = parser.parse_args()
    x = np.load(a.snapshot)
    h = x["history"].astype(float)
    mass = np.r_[0.0, x["body_mass"].astype(float)]
    inertias = np.concatenate([np.zeros((1, 3, 3)), x["body_inertia"].astype(float)])
    rows = []
    for event_index, event in enumerate(h):
        v0 = event[:, :3]
        w0 = event[:, 3:6]
        p = event[:, 6:9]
        q = event[:, 9:13]
        v1 = event[:, 13:16]
        w1 = event[:, 16:19]
        force = event[:, 19:22]
        torque = event[:, 22:25]
        dynamic = np.array([1, 2])
        dp = mass[:, None] * (v1 - v0)
        angular = np.zeros_like(dp)
        spin = np.zeros_like(dp)
        ke0 = 0.0
        ke1 = 0.0
        for b in dynamic:
            r = rotation(q[b])
            iw = r @ inertias[b] @ r.T
            spin[b] = iw @ (w1[b] - w0[b])
            angular[b] = spin[b] + np.cross(p[b], dp[b])
            ke0 += 0.5 * (mass[b] * np.dot(v0[b], v0[b]) + w0[b] @ iw @ w0[b])
            ke1 += 0.5 * (mass[b] * np.dot(v1[b], v1[b]) + w1[b] @ iw @ w1[b])
        contact_torque = torque + np.cross(p, force)
        linear_error = dp[dynamic].sum(axis=0) - force[dynamic].sum(axis=0)
        angular_error = angular[dynamic].sum(axis=0) - contact_torque[dynamic].sum(axis=0)
        contact_work = np.sum(force[dynamic] * (v0[dynamic] + v1[dynamic]) * 0.5) + np.sum(
            torque[dynamic] * (w0[dynamic] + w1[dynamic]) * 0.5
        )
        rows.append(
            {
                "linear_response_error_Ns": linear_error.tolist(),
                "angular_response_error_Nms": angular_error.tolist(),
                "independent_anchor_pair_angular_defect_Nms": contact_torque.sum(axis=0).tolist(),
                "delta_kinetic_energy_J": ke1 - ke0,
                "external_friction_midpoint_work_J": float(contact_work),
                "unresolved_joint_response_work_J": ke1 - ke0 - float(contact_work),
            }
        )
        if "joint_history" in x.files:
            joint = x["joint_history"][event_index].astype(float)
            jacobian = np.c_[joint[:, 2:8], joint[:, 8:14]]
            is_dynamic = joint[:, 14] != 0
            delta_lambda = joint[:, 1] - joint[:, 0]
            measured = np.c_[dp[dynamic], spin[dynamic]].ravel()
            external = np.c_[force[dynamic], torque[dynamic]].ravel()
            residual = measured - external - jacobian[is_dynamic].T @ delta_lambda[is_dynamic]
            hard_matrix = jacobian[~is_dynamic].T
            hard_lambda, _, rank, singular = np.linalg.lstsq(hard_matrix, residual, rcond=0)
            assert rank == 5, (rank, singular)
            reaction_error = residual - hard_matrix @ hard_lambda
            midpoint = np.c_[(v0[dynamic] + v1[dynamic]) * 0.5, (w0[dynamic] + w1[dynamic]) * 0.5].ravel()
            drive_work = float(delta_lambda[is_dynamic] @ (jacobian[is_dynamic] @ midpoint))
            hard_work = float(hard_lambda @ (jacobian[~is_dynamic] @ midpoint))
            rows[-1].update(
                actual_drive_reaction_work_J=drive_work,
                recovered_hard_reaction_work_J=hard_work,
                reaction_fit_error_mixed=float(np.max(np.abs(reaction_error))),
                full_work_budget_error_J=ke1 - ke0 - float(contact_work) - drive_work - hard_work,
            )
    report = {
        "scope": "Offline normalized-pose inertia reconstruction; first/last128actualfrictionphases. Native compliant-joint response may do work; when captured, actual drive delta-lambda is used and hard reaction is recovered with rank5 and residual checks. This is not an independently captured hard-joint impulse budget. Independent anchor pair angular defect is exposed, never assumed zero.",
        "events_saved": len(rows),
        "events_total": int(x["event_count"][0]),
        "patch_count": x["patch_count"].tolist(),
        "unsupported_error": x["error"].tolist(),
        "max_linear_response_error_Ns": max(np.linalg.norm(r["linear_response_error_Ns"]) for r in rows),
        "max_angular_response_error_Nms": max(np.linalg.norm(r["angular_response_error_Nms"]) for r in rows),
        "max_independent_anchor_angular_defect_Nms": max(
            np.linalg.norm(r["independent_anchor_pair_angular_defect_Nms"]) for r in rows
        ),
        "max_positive_friction_midpoint_work_J": max(r["external_friction_midpoint_work_J"] for r in rows),
        "rows": rows,
    }
    if "joint_history" in x.files:
        report["max_reaction_fit_error_mixed"] = max(r["reaction_fit_error_mixed"] for r in rows)
        report["max_full_work_budget_error_J"] = max(abs(r["full_work_budget_error_J"]) for r in rows)
        report["joint_work_method"] = (
            "Actual captured compliant-drive delta-lambda, plus five independent hard reaction impulses recovered from remaining physical impulse; no physical mode removed (rank5 asserted)."
        )
    assert np.all(np.isfinite(h)) and not np.any(x["error"])
    a.snapshot.with_suffix(".ledger.json").write_text(json.dumps(report, indent=2))
    print({k: v for k, v in report.items() if k != "rows"})


if __name__ == "__main__":
    main()
