"""Homogeneous whole-assembly joint response for a frozen support correction."""

import json
from pathlib import Path

import numpy as np

from local_studies.colibri.coulomb_semismooth import natural_map_evaluator


def main():
    """Preserve all hard-row velocities and soft-row constitutive residuals."""
    import scipy.linalg as la
    import sympy as sp
    from scipy.optimize import least_squares

    a = np.load("/tmp/colibri_total_normal_support_phases330.npz")
    d = {k.split(".", 1)[1]: a[k] for k in a.files if k.startswith("biased_solved.")}
    local = np.load("/tmp/colibri_total_normal_support_frozen.npz")
    report_local = json.load(open("/tmp/colibri_total_normal_support_frozen.json"))
    bodies = np.flatnonzero(d["inverse_mass"] > 0)
    index = {b: i for i, b in enumerate(bodies)}
    size = len(bodies) * 6
    W = np.zeros((size, size))
    u = np.zeros(size)
    for b, i in index.items():
        xx, yy, zz, xy, xz, yz = d["inverse_inertia"][b].astype(float)
        W[6 * i : 6 * i + 3, 6 * i : 6 * i + 3] = np.eye(3) * d["inverse_mass"][b]
        W[6 * i + 3 : 6 * i + 6, 6 * i + 3 : 6 * i + 6] = [[xx, xy, xz], [xy, yy, yz], [xz, yz, zz]]
        start, end = int(d["copy_section_end"][b - 1]), int(d["copy_section_end"][b])
        u[6 * i : 6 * i + 6] = np.r_[
            d["copy_velocity"][start:end].astype(float).mean(0),
            d["copy_angular_velocity"][start:end].astype(float).mean(0),
        ]
    C = np.zeros((len(local["C"]), size))
    for i, b in enumerate(report_local["bodies"]):
        C[:, 6 * index[b] : 6 * index[b] + 6] = local["C"][:, 6 * i : 6 * i + 6]
    jd = d["joint_data"].view(np.int32)
    native = []
    exact = []
    soft = []
    compliance = []
    def R(x):
        return sp.Rational(float(x))
    for j in range(int(a["num_joints"][0])):
        st = int(d["joint_structural_index"][j])
        rows = d["joint_row_indices"][j, : int(d["joint_row_count"][j])]
        joint_rows = []
        for row in rows:
            loc = int(d["joint_row_local"][row])
            v = np.zeros(size)
            for b, key in zip(jd[1:3, j], ("joint_wrench0", "joint_wrench1"), strict=True):
                if b in index:
                    v[6 * index[b] : 6 * index[b] + 6] = d[key][st, loc]
            joint_rows.append(v)
        point_rows = [k for k, v in enumerate(joint_rows) if np.any(v.reshape(-1, 6)[:, :3])]
        pivot = None
        if point_rows:
            assert len(point_rows) == 3
            ends = [int(b) for b in jd[1:3, j] if b in index]
            assert len(ends) == 2
            inferred = []
            for b in ends:
                f = np.array([joint_rows[k][6 * index[b] : 6 * index[b] + 3] for k in point_rows])
                t = np.array([joint_rows[k][6 * index[b] + 3 : 6 * index[b] + 6] for k in point_rows])
                np.testing.assert_array_equal(f @ f.T, np.eye(3))
                inferred.append(d["position"][b].astype(float) + np.cross(f, t).sum(0) * 0.5)
            pivot = [(R(x) + R(y)) / 2 for x, y in zip(*inferred, strict=True)]
        for k, (row, v) in enumerate(zip(rows, joint_rows, strict=True)):
            symbolic = list(map(R, v))
            if k in point_rows:
                for b in ends:
                    i = index[b]
                    lever = sp.Matrix([pivot[z] - R(d["position"][b, z]) for z in range(3)])
                    force = sp.Matrix(symbolic[6 * i : 6 * i + 3])
                    symbolic[6 * i + 3 : 6 * i + 6] = list(lever.cross(force))
            native.append(v)
            exact.append(symbolic)
            soft.append(bool(d["joint_row_dynamic"][row]))
            compliance.append(1 / float(d["joint_dynamic_mass"][row]) if soft[-1] else 0)
    native = np.array(native)
    soft = np.array(soft)
    compliance = np.array(compliance)
    J = np.array(exact, dtype=float)
    hard_ids = np.flatnonzero(~soft)
    # Exact rational rank/pivots of reconstructed physical primitives, not an SVD cutoff.
    exact_hard = sp.Matrix([exact[i] for i in hard_ids])
    dm = sp.polys.matrices.DomainMatrix.from_Matrix(exact_hard.T)
    _, pivots = dm.rref()
    keep = np.array(pivots, dtype=int)
    print("EXACT_RANK", len(keep), "ROWS", len(hard_ids), flush=True)
    B = J[hard_ids[keep]]
    L = np.linalg.cholesky(W)
    F = B @ L
    scales = np.linalg.norm(F, axis=1)
    Q, T = la.qr((F / scales[:, None]).T, mode="full")
    rank = len(keep)
    Z = Q[:, rank:]
    response = L @ Z
    hard_covariance_error = float(np.max(np.abs(J[~soft] @ response)))
    S = J[soft]
    D = compliance[soft]
    H = np.eye(response.shape[1]) + (S @ response).T @ ((S @ response) / D[:, None])
    response = la.solve_triangular(np.linalg.cholesky(H), response.T, lower=True).T
    A = (C @ response) @ (C @ response).T
    old = local["initial"]
    mu = local["mu"]
    reg = local["regularization"]
    points = report_local["points"]
    free = u - response @ (response.T @ C.T @ old)
    rhs = C @ free + d["derived"][3:6, points].T.astype(float).ravel()
    evaluate, _, _, _ = natural_map_evaluator(A, rhs, reg, mu)
    result = least_squares(
        lambda x: evaluate(x)[0] * 1e4,
        old,
        jac=lambda x: evaluate(x)[1] * 1e4,
        max_nfev=120,
        ftol=1e-12,
        xtol=1e-12,
        gtol=1e-12,
    )
    lam = result.x
    delta = lam - old
    dv = response @ (response.T @ C.T @ delta)
    v = u + dv
    ds = -(S @ dv) / D
    # Recover a valid hard reaction using the certified independent row basis.
    remaining = np.linalg.solve(L, dv) - L.T @ (C.T @ delta + S.T @ ds)
    beta_scaled = la.solve_triangular(T[:rank, :rank], Q[:, :rank].T @ remaining, lower=False)
    beta = beta_scaled / scales
    impulse = C.T @ delta + S.T @ ds + B.T @ beta
    reconstruction = float(np.max(np.abs(W @ impulse - dv)))
    linear = np.zeros(3)
    angular = np.zeros(3)
    for b, i in index.items():
        p = impulse[6 * i : 6 * i + 6]
        linear += p[:3]
        angular += p[3:] + np.cross(d["position"][b], p[:3])
    # Ground is the opposite common-point reaction for every support impulse increment.
    ground = -(C.T @ delta).reshape(-1, 6)
    for b, i in index.items():
        linear += ground[i, :3]
        angular += ground[i, 3:] + np.cross(d["position"][b], ground[i, :3])
    midpoint = (u + v) * 0.5
    M = np.linalg.inv(W)
    energy = 0.5 * (v @ M @ v - u @ M @ u)
    work = midpoint @ impulse
    external = []
    residual_before = []
    residual_after = []
    h = d["headers"].view(np.int32)
    for col in range(int(d["column_count"][0])):
        b0, b1 = h[1:3, col]
        if {b0, b1} == {0, 1}:
            continue
        for p in range(h[5, col], h[5, col] + h[6, col]):
            n, t = d["lambdas"][:3, p].astype(float), d["lambdas"][3:6, p].astype(float)
            axes = np.array([n, t, np.cross(n, t)])
            jac = np.zeros((3, size))
            for b, sign, offset in ((b0, -1, 9), (b1, 1, 12)):
                if b in index:
                    i = index[b]
                    jac[:, 6 * i : 6 * i + 6] = sign * np.c_[axes, np.cross(d["derived"][offset : offset + 3, p], axes)]
            external.append((int(p), float(np.max(np.abs(jac @ dv)))))
            bias = d["derived"][3:6, p].astype(float)
            gamma = 0.0 if bias[0] > 0 else 0.05829954519867897 / (0.9417003989219666 * float(d["derived"][0, p]))
            friction = float(d["headers"][3, col]) if bias[0] <= float(a["dt"][0]) ** -1 * 0.002 else 0.0
            K = jac @ W @ jac.T
            K[0, 0] += gamma
            scale = max(np.diag(K))
            if scale == 0.0:
                assert not np.any(jac), "Unexpected zero mobility on dynamic external contact"
                continue
            old_lam = d["impulses"][:, p].astype(float)
            for vel, collection in ((u, residual_before), (v, residual_after)):
                gradient = jac @ vel + bias
                gradient[0] += gamma * old_lam[0]
                trial = old_lam - gradient / scale
                radius = friction * max(old_lam[0], 0.0)
                projection = trial[1:] * min(1.0, radius / max(np.linalg.norm(trial[1:]), 1e-300))
                normal = gradient[0] if trial[0] > 0 else scale * old_lam[0]
                collection.append(
                    (int(p), float(abs(normal)), float(np.linalg.norm(scale * (old_lam[1:] - projection))))
                )
    residual = float(np.max(np.abs(evaluate(lam)[0])))
    actual_impulse = np.linalg.solve(W, dv).reshape(-1, 6)
    actual_linear = (actual_impulse + ground)[:, :3].sum(0)
    actual_angular = sum(
        (
            actual_impulse[i, 3:] + ground[i, 3:] + np.cross(d["position"][b], actual_impulse[i, :3] + ground[i, :3])
            for b, i in index.items()
        ),
        np.zeros(3),
    )
    out = {
        "actual_momentum_error": np.r_[actual_linear, actual_angular].tolist(),
        "external_residual_before_max": np.max(np.array(residual_before)[:, 1:], axis=0).tolist(),
        "external_residual_after_max": np.max(np.array(residual_after)[:, 1:], axis=0).tolist(),
        "physical_intervention_accepted": False,
        "rejection": "Base remains moving; external contact residuals require global solve; FP64 reaction reconstruction not within1e-8",
        "scope": "Frozen homogeneous all-hard-joint correction; exact rational reconstructed pivot rank; all support points and actual compliant response; external impulses held fixed",
        "exact_rank": rank,
        "hard_rows": int(sum(~soft)),
        "null_dofs": int(Z.shape[1]),
        "normalized_factor_condition": float(np.linalg.cond(F / scales[:, None])),
        "geometry_reform_max": float(np.max(abs(J - native))),
        "hard_factor_residual": hard_covariance_error,
        "contact_residual": residual,
        "root_accepted": bool(residual < 1e-8),
        "nfev": result.nfev,
        "njev": result.njev,
        "hard_velocity_change": float(np.max(abs(J[~soft] @ dv))),
        "native_hard_velocity_change": float(np.max(abs(native[~soft] @ dv))),
        "soft_constitutive_change": float(np.max(abs(S @ dv + D * ds))),
        "reaction_reconstruction_error": reconstruction,
        "base_before": u[6 * index[1] : 6 * index[1] + 6].tolist(),
        "base_after": v[6 * index[1] : 6 * index[1] + 6].tolist(),
        "momentum_error": np.r_[linear, angular].tolist(),
        "energy_change_J": float(energy),
        "work_J": float(work),
        "work_error_J": float(energy - work),
        "max_external_contact_velocity_change": max(e[1] for e in external),
        "worst_external_contacts": sorted(external, key=lambda e: -e[1])[:10],
    }
    Path("/tmp/colibri_global_support_frozen.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))
    np.savez(
        "/tmp/colibri_global_support_frozen.npz",
        J=J,
        native_J=native,
        W=W,
        response=response,
        C=C,
        old=old,
        solution=lam,
        u=u,
        v=v,
        delta_joint_hard=beta,
        delta_joint_soft=ds,
        hard_keep=hard_ids[keep],
        soft=soft,
    )


if __name__ == "__main__":
    main()
