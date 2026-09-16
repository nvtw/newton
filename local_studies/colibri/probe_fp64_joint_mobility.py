"""Compare FP64 unsquared mobility with the certified high-precision reference."""

import json
from pathlib import Path

import numpy as np
import sympy as sp


def main():
    import scipy.linalg as la

    p = np.load("/tmp/colibri_joint_reformed_input.npz")
    m = np.load("/tmp/colibri_joint_constrained_mobility.npz")
    c = np.load("/tmp/colibri_joint_constrained_sweeps.npz")
    ref = np.load("/tmp/colibri_joint_constrained.npz")
    active = m["active"]
    R = p["compliance"]
    drives = m["drive_rows"]
    J = p["J"][:, active, :].reshape(188, -1)
    F = np.concatenate((J.T, np.eye(188)[R > 0]), axis=0)
    exact = sp.polys.matrices.DomainMatrix.from_Matrix(sp.Matrix([[sp.Rational(float(x)) for x in row] for row in F]))
    keep = list(exact.rref()[1])
    n = len(keep)
    assert n == 184
    L = la.block_diag(*(la.cholesky(p["W"][b], lower=True) for b in active))
    B = J @ L
    C = np.column_stack((B, np.diag(np.sqrt(R))[:, drives]))
    scale = 1 / np.linalg.norm(C[keep], axis=1)
    Q, T = la.qr((C[keep] * scale[:, None]).T, mode="full")
    Z = Q[:, n:]
    G = L @ Z[: len(L)]
    slack = Z[len(L) :]
    a = la.solve_triangular(T[:n, :n].T, -p["rhs"][keep] * scale, lower=True)
    z = Q[:, :n] @ a
    ell = la.solve_triangular(T[:n, :n], a)
    lam = np.zeros(188)
    lam[keep] = scale * ell
    dv = L @ z[: len(L)]
    contact = c["C"].reshape(-1, len(L))
    X = contact @ G
    Xref = contact @ m["G"]
    H = X @ X.T
    Href = Xref @ Xref.T
    body = G @ X.T
    bodyref = m["G"] @ Xref.T
    drive = (slack @ X.T) / np.sqrt(R[drives])[:, None]
    driveref = (m["slack"] @ Xref.T) / np.sqrt(R[drives])[:, None]
    residual = J @ body
    residual[drives] += R[drives, None] * drive
    baseline_residual = J @ dv + R * lam + p["rhs"]
    factorresponse = L.T @ J.T @ lam - z[: len(L)]
    masses = la.block_diag(*(np.linalg.inv(p["W"][b]) for b in active))
    physical = (masses @ body).reshape(len(active), 6, -1)
    f = contact.T.reshape(len(active), 6, -1)
    defect = physical - f
    angular = np.sum(
        defect[:, 3:]
        + np.cross(
            np.load("/tmp/colibri_support_relax330.npz")["position"][active].astype(float)[:, :, None],
            defect[:, :3],
            axisa=1,
            axisb=1,
            axisc=1,
        ),
        axis=0,
    )
    linear = defect[:, :3].sum(axis=0)
    work = np.sum(contact.T * body, axis=0)
    kin = np.sum(body * (masses @ body), axis=0)
    drivepenalty = np.sum(R[drives, None] * drive**2, axis=0)
    out = {
        "independent_rows": n,
        "max_contact_mobility_error": float(abs(H - Href).max()),
        "relative_contact_mobility_frobenius_error": float(np.linalg.norm(H - Href) / np.linalg.norm(Href)),
        "max_contact_body_response_error": float(abs(body - bodyref).max()),
        "max_drive_response_error": float(abs(drive - driveref).max()),
        "max_joint_response_residual_per_unit_contact_impulse": float(abs(residual).max()),
        "baseline_equation_residual": float(abs(baseline_residual).max()),
        "baseline_factor_response_residual": float(abs(factorresponse).max()),
        "baseline_max_dv_difference": float(abs(dv - ref["dv"].ravel()).max()),
        "max_P_error_per_unit_contact": float(abs(linear).max()),
        "max_L_error_per_unit_contact": float(abs(angular).max()),
        "max_relative_work_identity_error": float(np.max(abs(work - kin - drivepenalty) / np.maximum(abs(work), 1))),
        "minimum_diagonal_mobility": float(np.diag(H).min()),
    }
    np.savez("/tmp/colibri_joint_fp64_mobility.npz", G=G, slack=slack, active=active, drive_rows=drives, R=R)
    np.savez("/tmp/colibri_joint_fp64_response.npz", delta=lam, dv=dv.reshape(-1, 6), active=active)
    Path("/tmp/colibri_joint_fp64_mobility.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
