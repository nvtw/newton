# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Audit FP32 storage of accepted FP64 roots; not a native scatter simulation."""

import json
from pathlib import Path

import numpy as np

from local_studies.colibri.coulomb_semismooth import natural_map_evaluator


def half_ulp(value):
    """Bound round-to-nearest storage using the larger adjacent FP32 spacing."""
    stored = value.astype(np.float32)
    above = np.nextafter(stored, np.float32(np.inf)).astype(float)
    below = np.nextafter(stored, np.float32(-np.inf)).astype(float)
    return 0.5 * np.maximum(above - stored.astype(float), stored.astype(float) - below)


def main():
    """Retain raw physical inputs and quantify storage error separately."""
    reports = json.loads(Path("/tmp/high_mass_consistent960_replay.json").read_text())
    results = {}
    for case, report in reports.items():
        if not report["accepted"]:
            continue
        d = dict(np.load("/tmp/high_mass_consistent960_" + case + ".npz"))
        raw = dict(np.load("/tmp/high_mass_window120_live_before_" + case + ".npz"))
        J, W, u, lam, old = (d[key] for key in ("J", "inverse_mass", "u", "solution", "initial"))
        rounded = lam.astype(np.float32).astype(float)
        impulse = J.T @ (rounded - old)
        exact_velocity = u + W @ impulse
        stored_velocity = exact_velocity.astype(np.float32).astype(float)
        storage_error = stored_velocity - exact_velocity
        storage_bound = half_ulp(exact_velocity)
        assert np.all(np.abs(storage_error) <= storage_bound + 1e-15)
        moving = raw["inverse_mass"] > 0
        M = np.zeros_like(W)
        for body in np.flatnonzero(moving):
            ids = slice(6 * body, 6 * body + 6)
            M[ids, ids] = np.linalg.inv(W[ids, ids])
        reaction = impulse.reshape(-1, 6).copy()
        reaction[moving] = (M @ (stored_velocity - u)).reshape(-1, 6)[moving]
        balance_p = reaction[:, :3].sum(0)
        balance_l = (reaction[:, 3:] + np.cross(raw["positions"], reaction[:, :3])).sum(0)
        wrench_bound = (np.abs(M) @ storage_bound).reshape(-1, 6)
        p_bound = wrench_bound[:, :3].sum(0)
        positions = np.abs(raw["positions"].astype(float))
        force_bound = wrench_bound[:, :3]
        l_bound = wrench_bound[:, 3:].sum(0)
        for axis in range(3):
            a, b = (axis + 1) % 3, (axis + 2) % 3
            l_bound[axis] += np.sum(positions[:, a] * force_bound[:, b] + positions[:, b] * force_bound[:, a])
        energy = 0.5 * (stored_velocity @ M @ stored_velocity - u @ M @ u)
        work = impulse @ ((u + stored_velocity) * 0.5)
        work_bound = 0.5 * storage_bound @ np.abs(M) @ np.abs(u + stored_velocity)
        realized_rhs = J @ stored_velocity - d["A"] @ rounded
        evaluate, scale, _, _ = natural_map_evaluator(d["A"], realized_rhs, d["normal_regularization"])
        residual = evaluate(rounded)[0]
        original_eval = natural_map_evaluator(d["A"], d["rhs"], d["normal_regularization"])[0]
        original_residual = original_eval(lam)[0]
        # The disk projection is nonexpansive in its argument and Lipschitz
        # in its radius. Bound impulse quantization and resulting velocities.
        lambda_bound = half_ulp(lam)
        velocity_bound = np.abs(W @ J.T) @ lambda_bound + storage_bound
        gradient_bound = np.abs(J) @ velocity_bound
        residual_bound = np.zeros_like(lam)
        for point in range(len(lam) // 3):
            n = 3 * point
            t = slice(n + 1, n + 3)
            residual_bound[n] = 2 * scale[n] * lambda_bound[n] + gradient_bound[n]
            residual_bound[t] = (
                2 * scale[n] * np.linalg.norm(lambda_bound[t])
                + np.linalg.norm(gradient_bound[t])
                + 0.5 * scale[n] * lambda_bound[n]
            )
        assert np.all(np.abs(balance_p) <= p_bound + 1e-12), case
        assert np.all(np.abs(balance_l) <= l_bound + 1e-12), case
        assert abs(energy - work) <= work_bound + 1e-12, case
        assert np.all(np.abs(residual - original_residual) <= residual_bound + 1e-11), case
        triples = rounded.reshape(-1, 3)
        original_triples = lam.reshape(-1, 3)
        cone = np.linalg.norm(triples[:, 1:], axis=1) - 0.5 * triples[:, 0]
        original_cone = np.linalg.norm(original_triples[:, 1:], axis=1) - 0.5 * original_triples[:, 0]
        lb = lambda_bound.reshape(-1, 3)
        cone_bound = np.linalg.norm(lb[:, 1:], axis=1) + 0.5 * lb[:, 0]
        assert np.all(np.abs(cone - original_cone) <= cone_bound + 1e-12), case
        results[case] = {
            "positive_friction_disk_violation_Ns": float(max(0, np.max(cone))),
            "friction_disk_rounding_change_bound_Ns": float(np.max(cone_bound)),
            "oracle_residual": float(np.max(np.abs(original_residual))),
            "stored_residual": float(np.max(np.abs(residual))),
            "residual_change_bound": float(np.max(residual_bound)),
            "linear_balance": float(np.max(np.abs(balance_p))),
            "linear_rounding_bound": float(np.max(p_bound)),
            "angular_balance": float(np.max(np.abs(balance_l))),
            "angular_rounding_bound": float(np.max(l_bound)),
            "work_error_J": float(abs(energy - work)),
            "work_rounding_bound_J": float(work_bound),
            "energy_change_J": float(energy),
        }
    summary = {
        "scope": "FP64 common-pivot accumulation with FP32 impulse and final velocity storage; not actual native scatter",
        "accepted_oracles": len(results),
        "stored_residual_above_1e_minus8": sum(v["stored_residual"] > 1e-8 for v in results.values()),
        "all_rounding_bounds_pass": True,
        "maxima": {key: max(v[key] for v in results.values()) for key in next(iter(results.values()))},
    }
    Path("/tmp/high_mass_float32_storage.json").write_text(
        json.dumps({"summary": summary, "cases": results}, indent=2) + "\n"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
