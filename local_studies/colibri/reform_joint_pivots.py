"""Reform saved Cartesian joint point rows at common inferred pivots."""

import argparse
import json
from pathlib import Path

import numpy as np
import sympy as sp


def main():
    """Compare original rows with a consistently reconstructed shared pivot."""
    import scipy.linalg as la

    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshot", default="/tmp/colibri_support_relax330.npz")
    parser.add_argument("--operator", default="/tmp/colibri_global_joint_probe.npz")
    parser.add_argument("--output", default="/tmp/colibri_joint_reformed_input")
    args = parser.parse_args()

    s = np.load(args.snapshot)
    d = dict(np.load(args.operator))
    old = d["J"]
    J = old.copy()
    pos = s["position"].astype(float)
    pivots = []
    for joint, count in enumerate(s["joint_row_count"]):
        rows = [int(r) for r in s["joint_row_indices"][joint, :count] if np.any(old[r, :, :3] != 0)]
        if not rows:
            continue
        assert len(rows) == 3
        bodies = np.flatnonzero(np.any(old[rows[0], :, :3] != 0, axis=1))
        levers = []
        for body in bodies:
            f = old[rows, body, :3]
            t = old[rows, body, 3:]
            assert np.array_equal(f @ f.T, np.eye(3))
            levers.append(np.cross(f, t).sum(axis=0) * 0.5)
        pivot = (pos[bodies[0]] + levers[0] + pos[bodies[1]] + levers[1]) * 0.5
        pivots.append(pivot)
        for row in rows:
            for body in bodies:
                J[row, body, 3:] = np.cross(pivot - pos[body], old[row, body, :3])
    deltaJ = J - old
    d["rhs"] = d["rhs"] + np.einsum("rbi,bi->r", deltaJ, d["initial"])
    d["J"] = J
    d["A"] = np.einsum("rbi,bij,sbj->rs", J, d["W"], J) + np.diag(d["compliance"])
    np.savez(args.output + ".npz", **d)
    active = np.flatnonzero(np.any(d["W"] != 0, axis=(1, 2)))
    L = la.block_diag(*(la.cholesky(d["W"][b], lower=True) for b in active))
    C = np.column_stack(
        (J[:, active, :].reshape(188, -1) @ L, np.diag(np.sqrt(d["compliance"]))[:, d["compliance"] > 0])
    )
    sv = la.svdvals(C / np.linalg.norm(C, axis=1)[:, None])
    F = np.concatenate((J.reshape(188, -1).T, np.eye(188)[d["compliance"] > 0]), axis=0)
    exact = sp.polys.matrices.DomainMatrix.from_Matrix(sp.Matrix([[sp.Rational(float(x)) for x in row] for row in F]))
    null = exact.nullspace().to_Matrix()
    rhs_errors = []
    for i in range(null.rows):
        n = list(null.row(i))
        norm = max(abs(x) for x in n)
        rhs_errors.append(float(sum(x * sp.Rational(float(y)) for x, y in zip(n, d["rhs"], strict=False)) / norm))
    torque = np.sum(J[:, :, 3:] + np.cross(pos[None, :, :], J[:, :, :3]), axis=1)
    out = {
        "max_operator_change": float(abs(deltaJ).max()),
        "max_rhs_change": float(abs(np.einsum("rbi,bi->r", deltaJ, d["initial"])).max()),
        "max_row_torque_defect": float(abs(torque).max()),
        "exact_nullity": null.rows,
        "normalized_null_rhs_errors": rhs_errors,
        "scaled_singular_tail": sv[-12:].tolist(),
    }
    Path(args.output + ".json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
