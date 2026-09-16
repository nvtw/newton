"""Certify one fixed joint row selection across actual captured poses."""

import argparse
import json
from pathlib import Path

import numpy as np
import sympy as sp


def exact_operator(d):
    J = d["J"].reshape(len(d["rhs"]), -1)
    F = np.concatenate((J.T, np.eye(len(J))[d["compliance"] > 0]), axis=0)
    return sp.polys.matrices.DomainMatrix.from_Matrix(sp.Matrix([[sp.Rational(float(x)) for x in row] for row in F]))


def main():
    import scipy.linalg as la

    parser = argparse.ArgumentParser()
    parser.add_argument("--operator", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    reference = np.load("/tmp/colibri_joint_reformed_input.npz")
    keep = list(exact_operator(reference).rref()[1])
    d = np.load(args.operator)
    exact = exact_operator(d)
    fullrank = len(exact.rref()[1])
    subsetrank = len(exact.extract(range(exact.shape[0]), keep).rref()[1])
    out = {
        "reference_rows": keep,
        "reference_rank": len(keep),
        "exact_pose_rank": fullrank,
        "exact_selected_rank": subsetrank,
        "selection_valid": fullrank == subsetrank == len(keep),
    }
    if out["selection_valid"]:
        active = np.flatnonzero(np.any(d["W"] != 0, axis=(1, 2)))
        J = d["J"][:, active, :].reshape(len(d["rhs"]), -1)
        R = d["compliance"]
        drives = np.flatnonzero(R > 0)
        L = la.block_diag(*(la.cholesky(d["W"][b], lower=True) for b in active))
        C = np.column_stack((J @ L, np.diag(np.sqrt(R))[:, drives]))
        scale = 1 / np.linalg.norm(C[keep], axis=1)
        Q, T = la.qr((C[keep] * scale[:, None]).T, mode="full")
        n = len(keep)
        a = la.solve_triangular(T[:n, :n].T, -d["rhs"][keep] * scale, lower=True)
        z = Q[:, :n] @ a
        lam = np.zeros(len(R))
        lam[keep] = scale * la.solve_triangular(T[:n, :n], a)
        dv = L @ z[: len(L)]
        residual = J @ dv + R * lam + d["rhs"]
        Z = Q[:, n:]
        G = L @ Z[: len(L)]
        slack = Z[len(L) :]
        check = J @ G
        check[drives] += np.sqrt(R[drives, None]) * slack
        sv = la.svdvals(C[keep] * scale[:, None])
        null = exact.nullspace().to_Matrix()
        errors = []
        for row in null.tolist():
            maximum = max(abs(x) for x in row)
            errors.append(float(sum(x * sp.Rational(float(y)) for x, y in zip(row, d["rhs"], strict=True)) / maximum))
        out.update(
            all_original_equations_max_residual=float(abs(residual).max()),
            complement_constraint_residual=float(abs(check).max()),
            smallest_independent_factor_sv=float(sv[-1]),
            max_impulse=float(abs(lam).max()),
            max_dv=float(np.linalg.norm(dv.reshape(-1, 6)[:, :3], axis=1).max()),
            max_dw=float(np.linalg.norm(dv.reshape(-1, 6)[:, 3:], axis=1).max()),
            exact_null_rhs_errors=errors,
        )
        np.savez(
            args.output + ".npz",
            G=G,
            slack=slack,
            delta=lam,
            dv=dv.reshape(-1, 6),
            active=active,
            drive_rows=drives,
            R=R,
        )
    Path(args.output + ".json").write_text(json.dumps(out, indent=2))
    print(json.dumps({k: v for k, v in out.items() if k != "reference_rows"}, indent=2))


if __name__ == "__main__":
    main()
