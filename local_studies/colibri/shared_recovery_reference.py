"""Frozen bias-only coherent patch projection; never modifies contact physics.

Offline FP64 analysis. Restrict numerical recovery to a chosen three-dimensional
translation/spin field while retaining every original physical response mode.
Zero-capacity points do not select the field. Failure leaves the original target.
"""

import argparse
import json
from pathlib import Path

import numpy as np


def project(points, tangents, normal, capacities, targets):
    weights = np.maximum(np.asarray(capacities), 0)
    if not np.any(weights > 0):
        return targets.copy(), {"status": "NO_LOAD"}
    center = np.average(points, axis=0, weights=weights)
    axis = np.eye(3)[np.argmin(abs(normal))]
    basis0 = np.cross(normal, axis)
    basis0 /= np.linalg.norm(basis0)
    basis1 = np.cross(normal, basis0)
    field = np.empty((len(points), 2, 3))
    field[:, :, 0] = tangents @ basis0
    field[:, :, 1] = tangents @ basis1
    field[:, :, 2] = np.einsum("nij,nj->ni", tangents, np.cross(normal, points - center))
    matrix = field.reshape(-1, 3)
    d = np.repeat(weights, 2)
    gram = matrix.T @ (d[:, None] * matrix)
    rhs = matrix.T @ (d * targets.ravel())
    try:
        factor = np.linalg.cholesky(gram)
        coefficients = np.linalg.solve(factor.T, np.linalg.solve(factor, rhs))
    except np.linalg.LinAlgError:
        return targets.copy(), {
            "status": "UNRESOLVED_FIELD",
            "action": "Original targets unchanged; no rank truncation",
        }
    projected = (matrix @ coefficients).reshape(-1, 2)
    error = targets - projected
    return projected, {
        "status": "PROJECTED_NUMERICAL_RECOVERY_ONLY",
        "coefficients": coefficients.tolist(),
        "gram_condition": float(np.linalg.cond(gram)),
        "normal_equation_residual": float(np.max(abs(matrix.T @ (d * error.ravel())))),
        "weighted_rms_change_m_s": float(np.sqrt(np.sum(weights[:, None] * error**2) / (2 * weights.sum()))),
        "max_target_change_m_s": float(np.max(abs(error))),
        "max_original_target_m_s": float(np.max(abs(targets))),
        "max_projected_target_m_s": float(np.max(abs(projected))),
    }


def fixtures():
    points = np.array([[-1.0, -1.0, 0.0], [1.0, -1.0, 0.0], [1.0, 1.0, 0.0], [-1.0, 1.0, 0.0]])
    tangents = np.broadcast_to(np.eye(3)[:2], (4, 2, 3)).copy()
    normal = np.array([0.0, 0.0, 1.0])
    coherent = np.array([0.3, -0.2]) + 0.1 * np.cross(normal, points)[:, :2]
    result, _ = project(points, tangents, normal, np.ones(4), coherent)
    np.testing.assert_allclose(result, coherent, rtol=0, atol=2e-16)
    result, _ = project(points, tangents, normal, np.ones(4), np.zeros((4, 2)))
    assert np.array_equal(result, np.zeros((4, 2)))
    original = coherent.copy()
    result, report = project(points, tangents, normal, np.zeros(4), original)
    assert report["status"] == "NO_LOAD" and np.array_equal(result, original)
    capacities = np.array([1.0, 1.0, 1.0, 0.0])
    changed = coherent.copy()
    changed[-1] = 100
    result, _ = project(points, tangents, normal, capacities, changed)
    np.testing.assert_allclose(result, coherent, rtol=0, atol=3e-16)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("snapshot", type=Path)
    args = parser.parse_args()
    fixtures()
    x = np.load(args.snapshot)
    columns = x["contact_columns_data"]
    headers = columns.view(np.int32)
    count = int(x["contact_valid_count"][0])
    data, derived, impulses = (x[k].astype(float) for k in ("contact_lambdas", "contact_derived", "contact_impulses"))
    groups = {}
    for col in range(headers.shape[1]):
        first, n = headers[5:7, col]
        if first < 0 or n <= 0 or first + n > count:
            continue
        pair = tuple(int(v) for v in headers[1:3, col])
        rows = groups.setdefault(pair, {})
        for k in range(first, first + n):
            rows[k] = float(columns[3, col])
    reports = []
    for pair, rows in groups.items():
        ids = np.array(sorted(rows))
        normals = data[:3, ids].T
        normal = normals[0] / np.linalg.norm(normals[0])
        assert np.max(abs(normals - normals[0])) == 0, "Bounded parallel-normal study only"
        tangents = np.stack([data[3:6, ids].T, np.cross(normals, data[3:6, ids].T)], axis=1)
        points = x["body_position"][pair[0]] + derived[9:12, ids].T
        capacities = np.array([rows[k] for k in ids]) * np.maximum(impulses[0, ids], 0)
        targets = derived[4:6, ids].T
        projected, report = project(points, tangents, normal, capacities, targets)
        report.update(
            pair=pair,
            points=ids.tolist(),
            capacities_Ns=capacities.tolist(),
            original_targets_m_s=targets.tolist(),
            projected_targets_m_s=projected.tolist(),
        )
        reports.append(report)
    output = args.snapshot.with_suffix(".shared_recovery.json")
    output.write_text(json.dumps({"scope": __doc__, "fixtures": "PASS", "patches": reports}, indent=2))
    print(
        [
            {
                k: v
                for k, v in r.items()
                if k not in ("points", "capacities_Ns", "original_targets_m_s", "projected_targets_m_s")
            }
            for r in reports
        ]
    )


if __name__ == "__main__":
    main()
