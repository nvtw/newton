"""Bounded original-order contact sweeps through a physical joint mobility."""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import sympy as sp


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--coupled-normals", action="store_true")
    parser.add_argument("--sweeps", type=int, default=8)
    parser.add_argument("--checkpoints", default="1,4,8")
    parser.add_argument("--mobility", default="/tmp/colibri_joint_constrained_mobility.npz")
    parser.add_argument("--baseline", default="/tmp/colibri_joint_constrained.npz")
    parser.add_argument("--output", default="/tmp/colibri_joint_constrained_sweeps")
    args = parser.parse_args()
    checkpoints = {int(x) for x in args.checkpoints.split(",")}
    s = np.load("/tmp/colibri_support_relax330.npz")
    p = np.load("/tmp/colibri_joint_reformed_input.npz")
    m = np.load(args.mobility)
    initial = np.load(args.baseline)
    active = m["active"]
    G = m["G"]
    slack = m["slack"]
    drives = m["drive_rows"]
    R = p["compliance"]
    J = p["J"][:, active, :].reshape(188, -1)
    h = s["headers"].view(np.int32)
    rows = []
    fullrows = []
    points = []
    friction = []
    for col in range(int(s["column_count"][0])):
        a, b = h[1:3, col]
        for point in range(h[5, col], h[5, col] + h[6, col]):
            if s["derived"][3, point] > 0:
                continue
            n = s["lambdas"][:3, point].astype(float)
            t = s["lambdas"][3:6, point].astype(float)
            axes = np.array([n, t, np.cross(n, t)])
            c = np.zeros((3, 38, 6))
            c[:, a] = -np.concatenate((axes, np.cross(s["derived"][9:12, point].astype(float), axes)), axis=1)
            c[:, b] = np.concatenate((axes, np.cross(s["derived"][12:15, point].astype(float), axes)), axis=1)
            rows.append(c[:, active].reshape(3, -1))
            fullrows.append(c)
            points.append(point)
            friction.append(float(s["headers"][4, col]))
    C = np.array(rows)
    CF = np.array(fullrows)
    physicalH = np.einsum("krbi,bij,ksbj->krs", CF, p["W"], CF)
    mu = np.array(friction)
    old = s["after_impulses"][:, points].T.astype(float)
    CG = C @ G
    H = CG @ CG.transpose(0, 2, 1)
    print("eligible", len(C), "normal mobility min/max", H[:, 0, 0].min(), H[:, 0, 0].max(), flush=True)
    hard = sp.polys.matrices.DomainMatrix.from_Matrix(
        sp.Matrix([[sp.Rational(float(x)) for x in row] for row in J[R == 0]])
    )
    N = hard.nullspace()
    exactC = sp.polys.matrices.DomainMatrix.from_Matrix(
        sp.Matrix([[sp.Rational(float(x)) for x in row] for row in C.reshape(-1, C.shape[-1])])
    )
    free = (exactC * N.transpose()).to_Matrix()
    normal_rank = []
    tangent_rank = []
    for k in range(len(C)):
        normal_rank.append(int(any(free[3 * k, j] != 0 for j in range(free.cols))))
        tangent_rank.append(int(free[3 * k + 1 : 3 * k + 3, :].rank()))
    print("exact ranks", np.bincount(normal_rank), np.bincount(tangent_rank), flush=True)
    eig = []
    U = []
    for k in range(len(C)):
        u, sig, _ = np.linalg.svd(CG[k, 1:], full_matrices=False)
        val = sig**2
        val[tangent_rank[k] :] = 0
        eig.append(val)
        U.append(u)
    eig = np.array(eig)
    U = np.array(U)
    lam = old.copy()
    base = p["initial"][active].ravel() + initial["dv"].ravel()
    coordinate = np.zeros(G.shape[1])
    M = np.zeros((len(active) * 6, len(active) * 6))
    for k, b in enumerate(active):
        M[6 * k : 6 * k + 6, 6 * k : 6 * k + 6] = np.linalg.inv(p["W"][b])
    native = p["initial"][active].ravel()
    base_joint = s["after_joint_accumulated"].astype(float) + initial["delta"]
    reports = []
    normal_reports = []
    if args.coupled_normals:
        import scipy.linalg as la
        from scipy.optimize import nnls

        qb, rb = la.qr(G, mode="economic")
        base_coordinates = la.solve_triangular(rb, qb.T @ base)
        assert np.max(np.abs(G @ base_coordinates - base)) < 1e-8
    shift_iterations = 0

    def disk(k, q, radius):
        nonlocal shift_iterations
        if radius <= 0:
            return np.zeros(2)
        values = eig[k]
        u = U[k]
        g = u.T @ q
        if np.all(values > 0):
            trial = -g / values
            if np.linalg.norm(trial) <= radius:
                return u @ trial
        elif np.all(g[values == 0] == 0):
            trial = np.zeros(2)
            positive = values > 0
            trial[positive] = -g[positive] / values[positive]
            if np.linalg.norm(trial) <= radius:
                return u @ trial
        lo = 0.0
        hi = max(np.linalg.norm(g) / radius, np.max(values), 1e-300)
        for _ in range(64):
            mid = (lo + hi) * 0.5
            length = np.linalg.norm(g / (values + mid))
            if length > radius:
                lo = mid
            else:
                hi = mid
            shift_iterations += 1
        return u @ (-g / (values + hi))

    def report(step, velocity, impulses, jointimp):
        g = (C @ velocity).reshape(-1, 3)
        normal = []
        tangent = []
        for k in range(len(C)):
            scale = float(np.max(np.diag(physicalH[k])))
            if scale == 0:
                scale = 1.0
            trial = impulses[k, 1:] - g[k, 1:] / scale
            radius = mu[k] * max(impulses[k, 0], 0)
            length = np.linalg.norm(trial)
            proj = trial * min(1, radius / max(length, 1e-300))
            normal.append(max(-g[k, 0], -impulses[k, 0] * scale, abs(min(impulses[k, 0] * scale, g[k, 0])), 0))
            tangent.append(np.linalg.norm(impulses[k, 1:] - proj) * scale)
        jr = J @ velocity + R * jointimp - s["joint_reference"]
        return {
            "sweep": step,
            "max_normal_residual": float(max(normal)),
            "max_tangent_residual": float(max(tangent)),
            "max_joint_residual": float(abs(jr).max()),
            "drive_residuals": jr[drives].tolist(),
            "max_impulse": float(abs(impulses).max()),
        }

    reports.append(report("native", native, old, s["after_joint_accumulated"]))
    reports.append(report(0, base, lam, base_joint))
    start = time.perf_counter()
    for sweep in range(1, args.sweeps + 1):
        if args.coupled_normals:
            without_normals = base_coordinates + coordinate - CG[:, 0].T @ lam[:, 0]
            proposed, objective_residual = nnls(CG[:, 0].T, -without_normals, maxiter=20 * len(C))
            normal_velocity = CG[:, 0] @ (without_normals + CG[:, 0].T @ proposed)
            error = max(
                float(np.max(-proposed)),
                float(np.max(-normal_velocity)),
                float(np.max(np.abs(np.minimum(proposed, normal_velocity)))),
            )
            normal_reports.append(
                {
                    "sweep": sweep,
                    "normal_kkt": error,
                    "nnls_objective_residual": objective_residual,
                    "active_normals": int(np.count_nonzero(proposed)),
                }
            )
            print("normalblock", normal_reports[-1], flush=True)
            if error > 1e-8:
                raise RuntimeError(f"Normal-block candidate failed original KKT: {error}")
            coordinate += CG[:, 0].T @ (proposed - lam[:, 0])
            lam[:, 0] = proposed
        for k in range(len(C)):
            g = C[k] @ base + CG[k] @ coordinate
            if normal_rank[k] and not args.coupled_normals:
                nextnormal = max(0.0, lam[k, 0] - g[0] / H[k, 0, 0])
                delta = nextnormal - lam[k, 0]
                lam[k, 0] = nextnormal
                coordinate += CG[k, 0] * delta
            g = C[k] @ base + CG[k] @ coordinate
            q = g[1:] - H[k, 1:, 1:] @ lam[k, 1:]
            target = disk(k, q, mu[k] * max(lam[k, 0], 0.0))
            delta = target - lam[k, 1:]
            lam[k, 1:] = target
            coordinate += CG[k, 1:].T @ delta
        if sweep in checkpoints:
            dv = G @ coordinate
            velocity = base + dv
            dl = (slack @ coordinate) / np.sqrt(R[drives])
            jointimp = base_joint.copy()
            jointimp[drives] += dl
            result = report(sweep, velocity, lam, jointimp)
            impulse = (M @ dv).reshape(-1, 6)
            result["deltaP"] = impulse[:, :3].sum(axis=0).tolist()
            result["deltaL"] = np.sum(
                impulse[:, 3:] + np.cross(s["position"][active].astype(float), impulse[:, :3]), axis=0
            ).tolist()
            fullforce = np.einsum("krbi,kr->bi", CF, lam - old)
            prescribed = np.setdiff1d(np.arange(38), active)
            reactionP = fullforce[prescribed, :3].sum(axis=0)
            reactionL = np.sum(
                fullforce[prescribed, 3:]
                + np.cross(s["position"][prescribed].astype(float), fullforce[prescribed, :3]),
                axis=0,
            )
            result["deltaP_plus_static_reaction"] = (np.array(result["deltaP"]) + reactionP).tolist()
            result["deltaL_plus_static_reaction"] = (np.array(result["deltaL"]) + reactionL).tolist()
            midpoint = base + dv * 0.5
            f = np.einsum("kri,kr->i", C, lam - old)
            cw = float(f @ midpoint)
            dw = float(dl @ (J[drives] @ midpoint))
            energy = float((velocity @ M @ velocity - base @ M @ base) * 0.5)
            result.update(deltaKE=energy, contact_work=cw, drive_work=dw, work_balance_error=abs(energy - cw - dw))
            reports.append(result)
            print(json.dumps(result), flush=True)
    output = {
        "coupled_normals": args.coupled_normals,
        "normal_block_reports": normal_reports,
        "eligible_points": len(C),
        "normal_rank_counts": np.bincount(normal_rank).tolist(),
        "tangent_rank_counts": np.bincount(tangent_rank).tolist(),
        "point_updates": args.sweeps * len(C),
        "disk_shift_iterations": shift_iterations,
        "seconds": time.perf_counter() - start,
        "reports": reports,
    }
    Path(args.output + ".json").write_text(json.dumps(output, indent=2))
    np.savez(
        args.output + ".npz",
        points=points,
        C=C,
        lam=lam,
        old=old,
        base=base,
        coordinate=coordinate,
        G=G,
        active=active,
        mu=mu,
    )
    print(json.dumps(output, indent=2), flush=True)


if __name__ == "__main__":
    main()
