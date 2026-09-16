"""Bounded frozen tail nullspace solve with all native witness rows retained."""

import json
from pathlib import Path

import numpy as np

from local_studies.colibri.coulomb_semismooth import solve_coulomb
from newton._src.solvers.phoenx.constraints import constraint_joint as schema


def main(
    snapshot="/tmp/colibri_support_relax330.npz",
    output="/tmp/colibri_tail_component_reference",
    operator="/tmp/colibri_tail_component_operator.npz",
    optimizer_scale=1.0,
    newton_fallback=False,
    friction_seed_ladder=False,
):
    d = np.load(snapshot)
    o = np.load(operator)
    W = o["W"]
    Jb = o["Jb"]
    R = o["R"]
    J = o["J"]
    bodies = o["bodies"]
    points = o["points"]
    active = o["active"]
    selected = (3 * active[:, None] + np.arange(3)).ravel()
    C = J[selected]
    A = o["A"]
    u = np.concatenate((d["velocity"][bodies], d["angular_velocity"][bodies]), axis=1).astype(float).ravel()
    old = d["impulses"][:, points].T.astype(float).ravel()
    initial = old[selected]
    h = d["headers"].view(np.int32)
    mu = {}
    for col in o["columns"]:
        assert d["headers"][3, col] == d["headers"][4, col]
        for p in range(h[5, col], h[5, col] + h[6, col]):
            mu[int(p)] = float(d["headers"][3, col])
    friction = np.array([mu[int(p)] for p in points[active]])
    free = u - W @ C.T @ initial
    L = np.linalg.cholesky(W)
    U, S, Vt = np.linalg.svd(Jb @ L, full_matrices=True)
    rank = len(Jb)
    assert np.min(S) > np.finfo(float).eps * max(Jb.shape) * S[0]
    vbar = free - L @ Vt[:rank].T @ ((U[:, :rank].T @ (Jb @ free)) / S[:rank])
    jd = d["joint_data"].view(np.int32)
    body_map = {int(b): i for i, b in enumerate(bodies)}
    soft = []
    soft_mass = []
    soft_old = []
    soft_reference = []
    for joint in o["joints"]:
        a, b = jd[[int(schema._OFF_BODY1), int(schema._OFF_BODY2)], joint]
        structural = int(d["joint_structural_index"][joint])
        for row in d["joint_row_indices"][joint, : d["joint_row_count"][joint]]:
            if not d["joint_row_dynamic"][row]:
                continue
            assert not d["joint_bounded_drive"][row]
            local = int(d["joint_row_local"][row])
            jac = np.zeros(len(u))
            for body, key in ((a, "joint_wrench0"), (b, "joint_wrench1")):
                if body in body_map:
                    jac[6 * body_map[body] : 6 * body_map[body] + 6] += d[key][structural, local]
            soft.append(jac)
            soft_mass.append(1.0 / float(d["joint_dynamic_mass"][row]))
            soft_old.append(float(d["joint_accumulated"][row]))
            soft_reference.append(float(d["joint_reference"][row]))
    Js = np.array(soft).reshape(-1, len(u))
    Ds = np.array(soft_mass)
    old_soft = np.array(soft_old)
    reference_soft = np.array(soft_reference)
    if len(Js):
        T = Js @ R
        H = np.eye(R.shape[1]) + T.T @ (T / Ds[:, None])
        vbar -= R @ np.linalg.solve(H, T.T @ ((Js @ vbar + Ds * old_soft - reference_soft) / Ds))
        R = np.linalg.solve(np.linalg.cholesky(H), R.T).T
        A = (C @ R) @ (C @ R).T
    rhs = C @ vbar
    solution, reports = solve_coulomb(A, rhs, initial, np.zeros(len(active)), friction, residual_scale=optimizer_scale)
    if newton_fallback:
        from local_studies.colibri.coulomb_semismooth import natural_map_evaluator
        from local_studies.colibri.polish_frame_crank_newton import polish

        evaluate, _, _, _ = natural_map_evaluator(A, rhs, np.zeros(len(active)), friction)
        if np.max(np.abs(evaluate(solution)[0])) >= 1e-8:
            solution, newton_history, error = polish(A, rhs, solution, friction)
            reports.append(
                {
                    "method": "bounded_newton_armijo",
                    "residual": error,
                    "steps": len([entry for entry in newton_history if "fraction" in entry]),
                    "residual_evaluations": len(newton_history)
                    + sum(entry.get("backtracks", -1) + 1 for entry in newton_history),
                    "history": newton_history,
                }
            )
    if friction_seed_ladder:
        from local_studies.colibri.coulomb_semismooth import natural_map_evaluator
        from local_studies.colibri.frame_crank_friction_continuation import friction_continuation

        evaluate, _, _, _ = natural_map_evaluator(A, rhs, np.zeros(len(active)), friction)
        if np.max(np.abs(evaluate(solution)[0])) >= 1e-8:
            solution, stages, error = friction_continuation(A, rhs, solution, friction)
            reports.append(
                {
                    "method": "fixed_auxiliary_friction_seeds",
                    "residual": error,
                    "stages": stages,
                    "original_friction_only": True,
                }
            )
    v = vbar + R @ (R.T @ C.T @ solution)
    delta_soft = (reference_soft - Js @ v) / Ds - old_soft if len(Js) else np.zeros(0)
    beta = -((U / S**2) @ U.T) @ (Jb @ free + Jb @ W @ (C.T @ solution + Js.T @ delta_soft))
    fcontact = C.T @ (solution - initial)
    fhard = Jb.T @ beta
    fdrive = Js.T @ delta_soft
    fjoint = fhard + fdrive
    reconstructed = u + W @ (fcontact + fjoint)
    g = A @ solution + rhs
    triples = solution.reshape(-1, 3)
    grad = g.reshape(-1, 3)
    scale = np.diag(A).reshape(-1, 3).max(axis=1)
    normal = float(
        max(
            np.max(-grad[:, 0]),
            np.max(-triples[:, 0] * scale),
            np.max(np.abs(np.minimum(triples[:, 0] * scale, grad[:, 0]))),
            0,
        )
    )
    tangent = []
    for lam, g, s, mu in zip(triples, grad, scale, friction, strict=True):
        trial = lam[1:] - g[1:] / s
        projected = trial * min(1.0, mu * max(lam[0], 0) / max(np.linalg.norm(trial), 1e-300))
        tangent.append(float(np.linalg.norm((lam[1:] - projected) * s)))
    full = old.copy()
    full[selected] = solution
    M = np.linalg.inv(W)
    momentum = (M @ (v - u)).reshape(-1, 6)
    P = np.zeros(6)
    for i, b in enumerate(bodies):
        P[:3] += momentum[i, :3]
        P[3:] += momentum[i, 3:] + np.cross(d["position"][b], momentum[i, :3])
    static_reaction = np.zeros(6)
    point_index = {int(p): i for i, p in enumerate(points)}
    for col in o["columns"]:
        a, b = h[1:3, col]
        for p in range(h[5, col], h[5, col] + h[6, col]):
            n = d["lambdas"][:3, p].astype(float)
            t = d["lambdas"][3:6, p].astype(float)
            axes = np.array([n, t, np.cross(n, t)])
            index = point_index[int(p)]
            delta = full.reshape(-1, 3)[index] - old.reshape(-1, 3)[index]
            for body, sign, lever in ((a, -1, d["derived"][9:12, p]), (b, 1, d["derived"][12:15, p])):
                if body in body_map:
                    continue
                assert np.all(d["velocity"][body] == 0) and np.all(d["angular_velocity"][body] == 0)
                f = sign * axes.T @ delta
                static_reaction[:3] += f
                static_reaction[3:] += np.cross(d["position"][body] + lever, f)
    momentum_balance = P + static_reaction
    delta_energy = 0.5 * (v @ M @ v - u @ M @ u)
    cw = fcontact @ ((u + v) * 0.5)
    jw = fjoint @ ((u + v) * 0.5)
    drive_residual = float(np.max(np.abs(Js @ v + Ds * (old_soft + delta_soft) - reference_soft))) if len(Js) else 0.0
    cone_violation = float(np.max(np.linalg.norm(triples[:, 1:], axis=1) - friction * triples[:, 0]))
    report = {
        "normal_residual_m_s": normal,
        "tangent_residual_m_s": max(tangent),
        "joint_residual": float(np.max(np.abs(Jb @ v))),
        "cone_violation": float(np.max(np.linalg.norm(triples[:, 1:], axis=1) - friction * triples[:, 0])),
        "reconstruction_error": float(np.max(np.abs(reconstructed - v))),
        "momentum_change": P.tolist(),
        "momentum_minus_static_reaction": momentum_balance.tolist(),
        "energy_change_J": float(delta_energy),
        "contact_work_J": float(cw),
        "joint_work_J": float(jw),
        "hard_joint_work_J": float(fhard @ ((u + v) * 0.5)),
        "drive_work_J": float(fdrive @ ((u + v) * 0.5)),
        "drive_equation_residual": float(np.max(np.abs(Js @ v + Ds * (old_soft + delta_soft) - reference_soft)))
        if len(Js)
        else 0.0,
        "work_balance_error_J": float(delta_energy - cw - jw),
        "solver_reports": reports,
        "accepted": bool(
            max(normal, *tangent, float(np.max(np.abs(Jb @ v))), drive_residual) < 1e-8
            and cone_violation < 1e-8
            and np.max(np.abs(momentum_balance)) < 1e-10
            and abs(delta_energy - cw - jw) < 1e-12
        ),
        "fixed_speculative_impulses_unchanged": bool(
            np.array_equal(
                full.reshape(-1, 3)[np.setdiff1d(np.arange(len(points)), active)],
                old.reshape(-1, 3)[np.setdiff1d(np.arange(len(points)), active)],
            )
        ),
    }
    # Full boundary audit: no outside row is silently treated as solved.
    native = np.concatenate((d["velocity"], d["angular_velocity"]), axis=1).astype(float)
    after = native.copy()
    after[bodies] = v.reshape(-1, 6)
    jd = d["joint_data"].view(np.int32)
    external_joints = []
    external_contacts = []
    for j in range(int(d["num_joints"][0])):
        a, b = jd[[int(schema._OFF_BODY1), int(schema._OFF_BODY2)], j]
        if j in o["joints"] or not set(bodies).intersection((a, b)):
            continue
        structural = int(d["joint_structural_index"][j])
        for row in d["joint_row_indices"][j, : d["joint_row_count"][j]]:
            local = int(d["joint_row_local"][row])
            ja = d["joint_wrench0"][structural, local]
            jb = d["joint_wrench1"][structural, local]
            dyn = bool(d["joint_row_dynamic"][row])
            offset = (
                float(d["joint_accumulated"][row]) / float(d["joint_dynamic_mass"][row])
                - float(d["joint_reference"][row])
                if dyn
                else 0.0
            )
            external_joints.append(
                {
                    "joint": j,
                    "row": int(row),
                    "dynamic": dyn,
                    "before": float(ja @ native[a] + jb @ native[b] + offset),
                    "after": float(ja @ after[a] + jb @ after[b] + offset),
                }
            )
    for col in range(int(d["column_count"][0])):
        a, b = h[1:3, col]
        if col in o["columns"] or not set(bodies).intersection((a, b)):
            continue
        for p in range(h[5, col], h[5, col] + h[6, col]):
            n = d["lambdas"][:3, p].astype(float)
            t = d["lambdas"][3:6, p].astype(float)
            axes = np.array([n, t, np.cross(n, t)])
            ja = -np.concatenate((axes, np.cross(d["derived"][9:12, p], axes)), axis=1)
            jb = np.concatenate((axes, np.cross(d["derived"][12:15, p], axes)), axis=1)
            external_contacts.append(
                {
                    "column": col,
                    "point": p,
                    "fixed_speculative": bool(d["derived"][3, p] > 0),
                    "before": (ja @ native[a] + jb @ native[b]).tolist(),
                    "after": (ja @ after[a] + jb @ after[b]).tolist(),
                    "impulse": d["impulses"][:, p].tolist(),
                }
            )
    report["external_joint_rows"] = external_joints
    report["external_contact_points"] = external_contacts
    Path(output + ".json").write_text(json.dumps(report, indent=2))
    np.savez_compressed(
        output + ".npz",
        J=J,
        Jb=Jb,
        Js=Js,
        delta_soft=delta_soft,
        active=active,
        u=u,
        v=v,
        solution=solution,
        initial=initial,
        full_impulses=full,
        beta=beta,
        bodies=bodies,
        points=points,
        A=A,
        rhs=rhs,
        friction=friction,
    )
    print(
        json.dumps(
            {k: v for k, v in report.items() if k not in ("external_joint_rows", "external_contact_points")}, indent=2
        )
    )


if __name__ == "__main__":
    main()
