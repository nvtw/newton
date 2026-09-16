"""Bounded physical support+joint correction at the actual total-normal biased phase."""

import argparse
import json
from pathlib import Path

import numpy as np

from local_studies.colibri.coulomb_semismooth import natural_map_evaluator


def main():
    """Replace all owned impulses once, retaining every support witness and actual drive."""
    from scipy.optimize import least_squares

    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("biased", "relax"), required=True)
    options = parser.parse_args()
    phase = options.phase
    output = "/tmp/colibri_two_body_full_" + phase
    source = Path("/tmp/colibri_base_frame_totalnormal_phases330.npz")
    arrays = np.load(source)
    d = {k.split(".", 1)[1]: arrays[k] for k in arrays.files if k.startswith(phase + "_solved.")}
    h, jd = d["headers"].view(np.int32), d["joint_data"].view(np.int32)
    support = 1
    joints = [j for j in range(int(arrays["num_joints"][0])) if support in jd[1:3, j]]
    bodies = sorted({int(b) for j in joints for b in jd[1:3, j] if d["inverse_mass"][b] > 0})
    index = {b: i for i, b in enumerate(bodies)}
    size = len(bodies) * 6
    W = np.zeros((size, size))
    velocity = np.zeros(size)
    for b, i in index.items():
        xx, yy, zz, xy, xz, yz = d["inverse_inertia"][b].astype(float)
        W[6 * i : 6 * i + 3, 6 * i : 6 * i + 3] = np.eye(3) * d["inverse_mass"][b]
        W[6 * i + 3 : 6 * i + 6, 6 * i + 3 : 6 * i + 6] = [[xx, xy, xz], [xy, yy, yz], [xz, yz, zz]]
        start, end = int(d["copy_section_end"][b - 1]), int(d["copy_section_end"][b])
        velocity[6 * i : 6 * i + 6] = np.r_[
            d["copy_velocity"][start:end].astype(float).mean(0),
            d["copy_angular_velocity"][start:end].astype(float).mean(0),
        ]
    B, diagonal, targets, old_joint = [], [], [], []
    for j in joints:
        st = int(d["joint_structural_index"][j])
        for row in d["joint_row_indices"][j, : int(d["joint_row_count"][j])]:
            local = int(d["joint_row_local"][row])
            jac = np.zeros(size)
            for body, key in zip(jd[1:3, j], ("joint_wrench0", "joint_wrench1"), strict=True):
                if body in index:
                    i = index[body]
                    jac[6 * i : 6 * i + 6] += d[key][st, local]
            dynamic = bool(d["joint_row_dynamic"][row])
            B.append(jac)
            diagonal.append(1 / float(d["joint_dynamic_mass"][row]) if dynamic else 0)
            targets.append(
                float(d["joint_reference"][row])
                if dynamic
                else (-float(d["joint_bias"][st, local]) if phase == "biased" else 0.0)
            )
            old_joint.append(float(d["joint_accumulated"][row]))
    B, diagonal, targets, old_joint = map(np.asarray, (B, diagonal, targets, old_joint))
    C, points, mu, old, bias, regularization = [], [], [], [], [], []
    ground_wrenches = []
    mc, ic = 0.9417003989219666, 0.05829954519867897
    for col in range(int(d["column_count"][0])):
        a, b = h[1:3, col]
        if 0 not in (a, b) or not (a in index or b in index):
            continue
        for p in range(h[5, col], h[5, col] + h[6, col]):
            if phase == "relax" and d["derived"][3, p] > 0:
                continue  # Native relax holds these exact captured impulses fixed.
            assert d["derived"][8, p] <= 0, "Explicit contact PD needs its own captured equation"
            assert d["headers"][3, col] == d["headers"][4, col], "Distinct static/dynamic cones require explicit branch"
            n, t = d["lambdas"][:3, p].astype(float), d["lambdas"][3:6, p].astype(float)
            axes = np.array([n, t, np.cross(n, t)])
            jac = np.zeros((3, size))
            ground = np.zeros((3, 6))
            for body, sign, start in ((a, -1, 9), (b, 1, 12)):
                wrench = sign * np.c_[axes, np.cross(d["derived"][start : start + 3, p].astype(float), axes)]
                if body in index:
                    i = index[body]
                    jac[:, 6 * i : 6 * i + 6] += wrench
                else:
                    ground += wrench
            C.extend(jac)
            ground_wrenches.append(ground)
            points.append(int(p))
            mu.append(float(d["headers"][3, col]))
            old.extend(d["impulses"][:, p].astype(float))
            bias.append(float(d["derived"][3, p]))
            regularization.append(
                0.0 if phase == "relax" or d["derived"][3, p] > 0 else ic / (mc * float(d["derived"][0, p]))
            )
            if d["derived"][3, p] > float(arrays["dt"][0]) ** -1 * 0.002:
                mu[-1] = 0.0
    C, old, mu, bias, regularization = map(np.asarray, (C, old, mu, bias, regularization))
    # Remove TOTAL owned impulse exactly once; retain captured unowned reactions.
    free = velocity - W @ (C.T @ old + B.T @ old_joint)
    K = B @ W @ B.T + np.diag(diagonal)
    vbar = free + W @ B.T @ np.linalg.solve(K, targets - B @ free)
    P = W - W @ B.T @ np.linalg.solve(K, B @ W)
    A = C @ P @ C.T
    rhs = C @ vbar
    if phase == "biased":
        rhs += d["derived"][3:6, points].T.astype(float).ravel()
    evaluate, _, _, _ = natural_map_evaluator(A, rhs, regularization, mu)
    initial_residual = float(np.max(np.abs(evaluate(old)[0])))
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
    joint = np.linalg.solve(K, targets - B @ free - B @ W @ C.T @ lam)
    updated = free + W @ (C.T @ lam + B.T @ joint)
    impulse = C.T @ (lam - old) + B.T @ (joint - old_joint)
    np.testing.assert_allclose(updated, velocity + W @ impulse, atol=1e-12, rtol=0)
    actual_gradient = C @ updated
    if phase == "biased":
        actual_gradient += d["derived"][3:6, points].T.astype(float).ravel()
    actual_gradient[::3] += regularization * lam[::3]
    residual = float(np.max(np.abs(evaluate(lam)[0])))
    triples = lam.reshape(-1, 3)
    physical = (C @ updated).reshape(-1, 3)
    ground = sum((g.T @ (triples[i] - old.reshape(-1, 3)[i]) for i, g in enumerate(ground_wrenches)), np.zeros(6))
    momentum = np.zeros(6)
    for b, i in index.items():
        p = impulse[6 * i : 6 * i + 6]
        momentum += np.r_[p[:3], p[3:] + np.cross(d["position"][b], p[:3])]
    momentum += np.r_[ground[:3], ground[3:] + np.cross(d["position"][0], ground[:3])]
    M = np.linalg.inv(W)
    delta_energy = 0.5 * (updated @ M @ updated - velocity @ M @ velocity)
    work = impulse @ ((updated + velocity) * 0.5)
    midpoint = (updated + velocity) * 0.5
    contact_delta = (lam - old).reshape(-1, 3)
    contact_normal = np.zeros_like(contact_delta)
    contact_normal[:, 0] = contact_delta[:, 0]
    contact_tangent = contact_delta - contact_normal
    work_parts = {
        "contact_normal": float(midpoint @ C.T @ contact_normal.ravel()),
        "contact_tangent": float(midpoint @ C.T @ contact_tangent.ravel()),
        "joint_hard": float(midpoint @ B[diagonal == 0].T @ (joint - old_joint)[diagonal == 0]),
        "joint_drive": float(midpoint @ B[diagonal > 0].T @ (joint - old_joint)[diagonal > 0]),
    }
    tangent_bias = d["derived"][4:6, points].T.astype(float)
    external = []
    for j in range(int(arrays["num_joints"][0])):
        if j in joints or not any(b in index for b in jd[1:3, j]):
            continue
        st = int(d["joint_structural_index"][j])
        for row in d["joint_row_indices"][j, : int(d["joint_row_count"][j])]:
            local = int(d["joint_row_local"][row])
            change = 0.0
            for body, key in zip(jd[1:3, j], ("joint_wrench0", "joint_wrench1"), strict=True):
                if body in index:
                    i = index[body]
                    change += float(d[key][st, local] @ (updated - velocity)[6 * i : 6 * i + 6])
            external.append({"joint": int(j), "row": int(row), "velocity_residual_change": float(change)})
    report = {
        "source": str(source),
        "phase": phase,
        "speculative_policy": "All points variable in biased phase; positive-bias points held at captured impulses in relax",
        "scope": "Frozen replacement, all support points, actual biased law and joint drive; unowned impulses held fixed, not skipped-row live replay",
        "bodies": bodies,
        "joints": joints,
        "points": points,
        "normal_regularization": "Captured ic/(mc*eff_n) for overlap; zero for positive-bias predictive rows; actual tangent recovery retained",
        "tangent_bias_max_m_s": float(np.linalg.norm(tangent_bias, axis=1).max()),
        "work_parts_J": work_parts,
        "external_joint_residual_changes": external,
        "initial_reduced_natural_residual": initial_residual,
        "final_natural_residual": residual,
        "accepted": bool(residual < 1e-8),
        "nfev": result.nfev,
        "njev": result.njev,
        "termination": str(result.message),
        "joint_residual": float(np.max(np.abs(B @ updated + diagonal * joint - targets))),
        "base_before": velocity[:6].tolist(),
        "base_after": updated[:6].tolist(),
        "contact_tangent_speed_max": float(np.linalg.norm(physical[:, 1:], axis=1).max()),
        "minimum_normal_impulse": float(triples[:, 0].min()),
        "cone_violation": float((np.linalg.norm(triples[:, 1:], axis=1) - mu * triples[:, 0]).max()),
        "momentum_error": momentum.tolist(),
        "energy_change_J": float(delta_energy),
        "midpoint_work_J": float(work),
        "work_error_J": float(delta_energy - work),
    }
    Path(output + ".json").write_text(json.dumps(report, indent=2))
    np.savez(
        output + ".npz",
        W=W,
        B=B,
        C=C,
        A=A,
        rhs=rhs,
        regularization=regularization,
        mu=mu,
        initial=old,
        solution=lam,
        velocity=velocity,
        updated=updated,
        joint=joint,
        old_joint=old_joint,
        diagonal=diagonal,
        targets=targets,
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
