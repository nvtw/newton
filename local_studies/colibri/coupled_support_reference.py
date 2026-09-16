"""Frozen mass-nullspace support/contact Coulomb reference; no live solver edits."""

import json
from pathlib import Path

import numpy as np

from local_studies.colibri.coulomb_semismooth import solve_coulomb
from newton._src.solvers.phoenx.body import inertia_sym6_unpack_np
from newton._src.solvers.phoenx.constraints import constraint_joint as schema


def main(
    snapshot="/tmp/colibri_support_relax330.npz", copy_counts=None, output="/tmp/colibri_coupled_support_reference"
):
    d = np.load(snapshot)
    h = d["headers"].view(np.int32)
    jd = d["joint_data"].view(np.int32)
    support = 1
    joints = [
        j for j in range(int(d["num_joints"][0])) if support in jd[[int(schema._OFF_BODY1), int(schema._OFF_BODY2)], j]
    ]
    body_ids = sorted(
        {
            int(b)
            for j in joints
            for b in jd[[int(schema._OFF_BODY1), int(schema._OFF_BODY2)], j]
            if d["inverse_mass"][b] > 0
        }
    )
    body_map = {b: i for i, b in enumerate(body_ids)}
    dofs = 6 * len(body_ids)
    W = np.zeros((dofs, dofs))
    inverse_inertia = inertia_sym6_unpack_np(d["inverse_inertia"]).astype(float)
    for b, i in body_map.items():
        W[6 * i : 6 * i + 3, 6 * i : 6 * i + 3] = np.eye(3) * d["inverse_mass"][b]
        W[6 * i + 3 : 6 * i + 6, 6 * i + 3 : 6 * i + 6] = inverse_inertia[b]
    Wphysical = W.copy()
    if copy_counts is not None:
        for body, index in body_map.items():
            W[6 * index : 6 * index + 6, 6 * index : 6 * index + 6] *= copy_counts[body]
    u = np.concatenate((d["velocity"][body_ids], d["angular_velocity"][body_ids]), axis=1).astype(float).ravel()
    rows = []
    soft_rows = []
    soft_d = []
    soft_old = []
    soft_reference = []
    for j in joints:
        structural = int(d["joint_structural_index"][j])
        a, b = jd[[int(schema._OFF_BODY1), int(schema._OFF_BODY2)], j]
        for index in range(int(d["joint_row_count"][j])):
            row = int(d["joint_row_indices"][j, index])
            dynamic = bool(d["joint_row_dynamic"][row])
            if d["joint_bounded_drive"][row]:
                raise ValueError("Finite drive bound requires a bounded local equation")
            local = int(d["joint_row_local"][row])
            jac = np.zeros(dofs)
            for body, key in ((a, "joint_wrench0"), (b, "joint_wrench1")):
                if body in body_map:
                    i = body_map[body]
                    jac[6 * i : 6 * i + 6] += d[key][structural, local]
            if dynamic:
                soft_rows.append(jac)
                soft_d.append(1.0 / float(d["joint_dynamic_mass"][row]))
                soft_old.append(float(d["joint_accumulated"][row]))
                soft_reference.append(float(d["joint_reference"][row]))
            else:
                rows.append(jac)
    Jb = np.asarray(rows)
    Js = np.asarray(soft_rows).reshape(-1, dofs)
    Ds = np.asarray(soft_d)
    old_soft = np.asarray(soft_old)
    reference_soft = np.asarray(soft_reference)
    points = []
    mus = []
    owners = []
    for col in range(int(d["column_count"][0])):
        a, b = h[1:3, col]
        if support not in (a, b) or not (d["inverse_mass"][a] == 0 or d["inverse_mass"][b] == 0):
            continue
        for k in range(h[5, col], h[5, col] + h[6, col]):
            points.append(int(k))
            owners.append((int(a), int(b)))
            mus.append(d["headers"][[3, 4], col])
    points = np.asarray(points)
    mus = np.asarray(mus)
    if not np.all(mus == mus[0, 0]):
        raise ValueError("Initial reference requires equal static/dynamic friction across selected manifold")
    mu = float(mus[0, 0])
    J = np.zeros((3 * len(points), dofs))
    common = []
    for p, (k, (a, b)) in enumerate(zip(points, owners, strict=True)):
        n = d["lambdas"][:3, k].astype(float)
        t = d["lambdas"][3:6, k].astype(float)
        r0 = d["derived"][9:12, k].astype(float)
        r1 = d["derived"][12:15, k].astype(float)
        common.append(float(np.linalg.norm(d["position"][a] + r0 - d["position"][b] - r1)))
        for axis, direction in enumerate((n, t, np.cross(n, t))):
            for body, sign, lever in ((a, -1, r0), (b, 1, r1)):
                if body in body_map:
                    i = body_map[body]
                    J[3 * p + axis, 6 * i : 6 * i + 3] += sign * direction
                    J[3 * p + axis, 6 * i + 3 : 6 * i + 6] += sign * np.cross(lever, direction)
    # Native relaxation freezes speculative rows entirely. Retain full arrays
    # and explicit fixed values, solving only the free impulse variables.
    speculative = d["derived"][3, points] > 0
    active = np.flatnonzero(~speculative)
    selected = np.concatenate([np.arange(3 * i, 3 * i + 3) for i in active])
    C = J[selected]
    old = d["impulses"][:, points].T.astype(float).ravel()
    initial = old[selected]
    free = u - W @ C.T @ initial
    L = np.linalg.cholesky(W)
    U, S, Vt = np.linalg.svd(Jb @ L, full_matrices=True)
    threshold = np.finfo(float).eps * max(Jb.shape) * S[0]
    rank = int(np.count_nonzero(S > threshold))
    Z = Vt[rank:].T
    R = L @ Z
    vbar = free - L @ Vt[:rank].T @ ((U[:, :rank].T @ (Jb @ free)) / S[:rank])
    R_hard = R.copy()
    if len(Js):
        T = Js @ R_hard
        H = np.eye(R_hard.shape[1]) + T.T @ (T / Ds[:, None])
        vbar = vbar - R_hard @ np.linalg.solve(H, T.T @ ((Js @ vbar + Ds * old_soft - reference_soft) / Ds))
        R = np.linalg.solve(np.linalg.cholesky(H), R_hard.T).T
    A = (C @ R) @ (C @ R).T
    pd = d["derived"][8, points[active]] > 0
    regularization = np.where(pd, d["derived"][6, points[active]], 0.0).astype(float)
    rhs = C @ vbar
    rhs[::3] -= np.where(pd, d["derived"][7, points[active]], 0.0)
    solution, reports = solve_coulomb(A, rhs, initial, regularization, friction=mu)
    full = old.copy()
    full[selected] = solution
    v = vbar + R @ (R.T @ C.T @ solution)
    Bpinv = (U[:, :rank] / S[:rank] ** 2) @ U[:, :rank].T
    delta_soft = (reference_soft - Js @ v) / Ds - old_soft if len(Js) else np.zeros(0)
    beta = -Bpinv @ (Jb @ free + Jb @ W @ (C.T @ solution + Js.T @ delta_soft))
    reconstructed = u + W @ (Jb.T @ beta + Js.T @ delta_soft + C.T @ (solution - initial))
    g = A @ solution + rhs
    g[::3] += regularization * solution[::3]
    triples = solution.reshape(-1, 3)
    gradient = g.reshape(-1, 3)
    scale = np.diag(A).reshape(-1, 3).max(axis=1)
    normal = max(
        float(np.max(np.maximum(-gradient[:, 0], 0))),
        float(np.max(np.maximum(-triples[:, 0] * scale, 0))),
        float(np.max(np.abs(np.minimum(triples[:, 0] * scale, gradient[:, 0])))),
    )
    tangent = []
    for row, grad, s in zip(triples, gradient, scale, strict=True):
        trial = row[1:] - grad[1:] / s
        radius = mu * max(row[0], 0)
        projected = trial * min(1.0, radius / max(np.linalg.norm(trial), 1e-300))
        tangent.append(float(np.linalg.norm((row[1:] - projected) * s)))
    cone = float(np.max(np.linalg.norm(triples[:, 1:], axis=1) - mu * triples[:, 0]))
    M = np.linalg.inv(W)
    dv = v - u
    external = np.zeros(6)
    change = np.zeros(6)
    joint_result = np.zeros(6)
    for b, i in body_map.items():
        momentum = M[6 * i : 6 * i + 6, 6 * i : 6 * i + 6] @ dv[6 * i : 6 * i + 6]
        change[:3] += momentum[:3]
        change[3:] += momentum[3:] + np.cross(d["position"][b], momentum[:3])
        jp = (Jb.T @ beta + Js.T @ delta_soft)[6 * i : 6 * i + 6]
        joint_result[:3] += jp[:3]
        joint_result[3:] += jp[3:] + np.cross(d["position"][b], jp[:3])
        cp = (C.T @ (solution - initial))[6 * i : 6 * i + 6]
        external[:3] += cp[:3]
        external[3:] += cp[3:] + np.cross(d["position"][b], cp[:3])
    energy_change = 0.5 * (v @ M @ v - u @ M @ u)
    contact_work = float((solution - initial) @ (C @ ((v + u) * 0.5)))
    hard_joint_work = float(beta @ (Jb @ ((v + u) * 0.5)))
    drive_work = float(delta_soft @ (Js @ ((v + u) * 0.5)))
    joint_work = hard_joint_work + drive_work
    drive_residual = float(np.max(np.abs(Js @ v + Ds * (old_soft + delta_soft) - reference_soft))) if len(Js) else 0.0
    report = {
        "stage": "Actual final330 pre-relax; native speculative impulses held fixed",
        "bodies": body_ids,
        "body_copy_counts": [1 if copy_counts is None else int(copy_counts[b]) for b in body_ids],
        "joints": joints,
        "bilateral_rows": len(Jb),
        "bilateral_rank": rank,
        "retained_points": points.tolist(),
        "free_points": points[active].tolist(),
        "fixed_speculative_points": points[speculative].tolist(),
        "null_dofs": Z.shape[1],
        "compliant_rows": len(Js),
        "drive_equation_residual": drive_residual,
        "drive_work_J": drive_work,
        "hard_joint_work_J": hard_joint_work,
        "common_point_max_error_m": max(common),
        "solver_reports": reports,
        "normal_residual_m_s": normal,
        "tangent_residual_m_s": max(tangent),
        "cone_violation_Ns": cone,
        "bilateral_velocity_residual": float(np.max(np.abs(Jb @ v))),
        "reaction_reconstruction_error": float(np.max(np.abs(v - reconstructed))),
        "momentum_minus_static_reaction": (change - external).tolist(),
        "joint_internal_wrench": joint_result.tolist(),
        "energy_change_J": float(energy_change),
        "contact_work_J": contact_work,
        "joint_work_J": joint_work,
        "work_balance_error_J": float(energy_change - contact_work - joint_work),
        "fixed_impulses_unchanged": bool(
            np.array_equal(full.reshape(-1, 3)[speculative], old.reshape(-1, 3)[speculative])
        ),
        "accepted": bool(
            max(normal, *tangent, 0.0, cone, drive_residual, float(np.max(np.abs(Jb @ v)))) < 1e-8
            and np.max(np.abs(change - external)) < 1e-10
            and abs(energy_change - contact_work - joint_work) < 1e-12
        ),
    }
    Path(output + ".json").write_text(json.dumps(report, indent=2))
    np.savez_compressed(
        output + ".npz",
        Wphysical=Wphysical,
        W=W,
        Jb=Jb,
        J=J,
        A=A,
        rhs=rhs,
        initial=initial,
        solution=solution,
        regularization=regularization,
        u=u,
        v=v,
        beta=beta,
        Js=Js,
        Ds=Ds,
        old_soft=old_soft,
        delta_soft=delta_soft,
        reference_soft=reference_soft,
        points=points,
        active=active,
        full_impulses=full,
    )
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
