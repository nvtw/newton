"""Exact original-row normal-circuit opening proposals, CPU diagnostic only."""

import json
import types
from fractions import Fraction
from pathlib import Path

import numpy as np

from local_studies.colibri import physical_face_active_set as frozen
from local_studies.colibri.coulomb_semismooth import natural_map_evaluator
from local_studies.colibri.physical_face_support import solve_face as original_face


def determinant3(m):
    """Evaluate a three-row determinant exactly on the supplied binary floats."""
    return (
        m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
        - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
        + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0])
    )


def certificate(a, active):
    """Certify a four-normal, three-component exact dependence without mode cuts."""
    points = np.flatnonzero(active & (a["gamma"] == 0))
    if len(points) != 4:
        return None
    rows = a["C"][3 * points]
    columns = np.flatnonzero(np.any(rows != 0, axis=0))
    if len(columns) != 3:
        return None
    matrix = [[Fraction.from_float(float(x)) for x in row] for row in rows[:, columns]]
    z = [(-1) ** i * determinant3([row for j, row in enumerate(matrix) if j != i]) for i in range(4)]
    if not any(z):
        return None
    assert all(sum(z[i] * matrix[i][j] for i in range(4)) == 0 for j in range(3))
    delta = sum(z[i] * Fraction.from_float(float(a["rhs"][3 * p])) for i, p in enumerate(points))
    if delta == 0:
        return None
    # If all normal gradients are nonnegative, z*g=delta forces at least one
    # gradient with z_i*delta>0 to be strictly positive: that normal must open.
    candidates = [int(p) for i, p in enumerate(points) if z[i] * delta > 0]
    zz = np.array([float(x) for x in z])
    norm = np.linalg.norm(zz)
    return {
        "points": points.tolist(),
        "original_nonzero_columns": columns.tolist(),
        "normalized_dependency": (zz / norm).tolist(),
        "normalized_target_defect": float(delta) / norm,
        "opening_candidates": candidates,
        "exact_binary_row_dependence": True,
    }


def solve_face(a, f, seed, active, stick):
    """Try only mathematically forced opening candidates; all rows remain checked."""
    proof = certificate(a, active)
    if proof is None:
        return original_face(a, f, seed, active, stick)
    evaluate, *_ = natural_map_evaluator(a["A"], a["rhs"], a["gamma"], a["mu"])
    best = None
    best_merit = np.inf
    records = []
    for point in proof["opening_candidates"][:3]:
        ac = active.copy()
        st = stick.copy()
        ac[point] = False
        st[point] = False
        trial, report = original_face(a, f, seed, ac, st)
        pivots = []
        for _ in range(3):
            bad = int(np.argmin(trial[::3]))
            if trial[3 * bad] >= -1e-12:
                break
            pivots.append({"point": bad, "negative_trial_normal": float(trial[3 * bad])})
            ac[bad] = False
            st[bad] = False
            trial, report = original_face(a, f, seed, ac, st)
        residual = evaluate(trial)[0]
        merit = float(residual @ residual)
        records.append(
            {
                "opened": point,
                "negative_normal_pivots": pivots,
                "face": report,
                "all_row_natural_error": float(np.max(abs(residual))),
            }
        )
        if merit < best_merit:
            best, best_merit = (trial, dict(report)), merit
        if np.max(abs(residual)) < 1e-10 and np.min(trial[::3]) >= -1e-12:
            break
    if best is None:
        return original_face(a, f, seed, active, stick)
    value, report = best
    report["exact_normal_circuit"] = proof
    report["opening_attempts"] = records
    return value, report


def solve(a):
    """Reuse the frozen outer safeguards verbatim with this additional face proposal."""
    namespace = dict(frozen.solve.__globals__)
    namespace["solve_face"] = solve_face
    function = types.FunctionType(frozen.solve.__code__, namespace)
    value, report = function(a)
    report["bounds"]["exact_circuit_opening_candidates"] = 3
    report["scope"] += "; exact binary original-row circuit opening, no constraint/mode deletion"
    return value, report


def main():
    a = dict(np.load("/tmp/colibri_two_body_adaptive_live60.rejected.npz"))
    value, report = solve(a)
    Path("/tmp/colibri_physical_face_circuit.json").write_text(json.dumps(report, indent=2))
    np.savez("/tmp/colibri_physical_face_circuit.npz", lam=value)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
