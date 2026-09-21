"""CPU frozen startup support capacity and observed base motion audit."""

import json
from pathlib import Path

import numpy as np
from scipy.optimize import linprog


def capture(path):
    a = np.load(path)
    h = a["headers"].view(np.int32)
    W = np.zeros((6, 6))
    W[:3, :3] = np.eye(3) * a["inverse_mass"][1]
    xx, yy, zz, xy, xz, yz = a["inverse_inertia"][1].astype(float)
    W[3:, 3:] = [[xx, xy, xz], [xy, yy, yz], [xz, yz, zz]]
    ids = []
    rows = []
    lams = []
    mus = []
    for c in range(int(a["column_count"][0])):
        b0, b1 = h[1:3, c]
        if {b0, b1} != {0, 1}:
            continue
        for k in range(h[5, c], h[5, c] + h[6, c]):
            if a["derived"][3, k] > 0:
                continue
            n = a["lambdas"][:3, k].astype(float)
            t = a["lambdas"][3:6, k].astype(float)
            axes = np.array([n, t, np.cross(n, t)])
            r = a["derived"][9:12, k] if b0 == 1 else a["derived"][12:15, k]
            rows.append(np.c_[axes, np.cross(r, axes)] * (-1 if b0 == 1 else 1))
            ids.append(k)
            lams.extend(a["after_impulses"][:, k])
            mus.append(a["headers"][3, c])
    J = np.vstack(rows)
    lam = np.array(lams, float)
    after = np.r_[a["after_velocity"][1], a["after_angular_velocity"][1]].astype(float)
    free = after - W @ J.T @ lam
    count = len(ids)
    rays = np.zeros((count * 3, count * 64))
    for i, mu in enumerate(mus):
        theta = np.arange(64) * 2 * np.pi / 64
        rays[3 * i, i * 64 : (i + 1) * 64] = 1
        rays[3 * i + 1, i * 64 : (i + 1) * 64] = mu * np.cos(theta)
        rays[3 * i + 2, i * 64 : (i + 1) * 64] = mu * np.sin(theta)
    B = W @ J.T @ rays * 1e-4
    scale = np.maximum(np.linalg.norm(B, axis=1), abs(free))
    lp = linprog(
        np.ones(count * 64),
        A_eq=B / scale[:, None],
        b_eq=-free / scale,
        bounds=(0, None),
        method="highs",
        options={"primal_feasibility_tolerance": 1e-10, "dual_feasibility_tolerance": 1e-10},
    )
    result = dict(
        source=str(path),
        points=ids,
        actual_after_twist=after.tolist(),
        free_twist=free.tolist(),
        cone_inner_polygon_feasible=bool(lp.success),
        status=lp.message,
    )
    if not lp.success:
        outer = rays.copy()
        outer[1::3] /= np.cos(np.pi / 64)
        outer[2::3] /= np.cos(np.pi / 64)
        base = W @ J.T * 1e-4 / scale[:, None]
        Bout = base @ outer
        target = -free / scale
        dual = linprog(
            np.zeros(6),
            A_ub=-Bout.T,
            b_ub=np.zeros(Bout.shape[1]),
            A_eq=target[None, :],
            b_eq=[-1.0],
            bounds=[(None, None)] * 6,
            method="highs",
        )
        result["outer_cone_separator_found"] = bool(dual.success)
        if dual.success:
            coeff = (base.T @ dual.x).reshape(-1, 3)
            exact_min = coeff[:, 0] - np.array(mus) * np.linalg.norm(coeff[:, 1:], axis=1)
            result["exact_circle_dual_min"] = float(min(exact_min))
            result["target_dual_dot"] = float(target @ dual.x)
            result["separator"] = dual.x.tolist()
            result["interpretation"] = (
                "Nonnegative exact circle dual and negative target dot certify fixed-reaction infeasibility, not fully coupled live motion necessity."
            )
    if lp.success:
        value = rays @ lp.x * 1e-4
        v = free + W @ J.T @ value
        t = value.reshape(-1, 3)
        result.update(
            velocity_residual=float(max(abs(v))),
            normal_sum_Ns=float(sum(t[:, 0])),
            circular_cone_violation=float(max(np.linalg.norm(t[:, 1:], axis=1) - np.array(mus) * t[:, 0])),
            active_points=[ids[i] for i in range(count) if t[i, 0] > 1e-12],
            required_wrench_Ns_Nms=(-np.linalg.solve(W, free)).tolist(),
        )
    return result


def main():
    a = np.load("/tmp/colibri_total_normal330.npz")
    body = int(np.flatnonzero(a["labels"] == "FrameGround")[0])
    q = a["q_history"][:, body]
    v = a["qd_history"][:, body]
    times = a["history_times"]
    dx = np.diff(np.vstack([a["initial_q"][body, :3], q[:, :3]]), axis=0)
    peak = np.argsort(np.linalg.norm(dx[:, :2], axis=1))[-8:][::-1]
    motion = dict(
        source="/tmp/colibri_total_normal330.npz",
        largest_frame_translations=[
            dict(time=float(times[i]), translation_m=dx[i].tolist(), twist=v[i].tolist()) for i in peak
        ],
        displacement_initial_to_first_m=(q[0, :3] - a["initial_q"][body, :3]).tolist(),
        displacement_initial_to_1s_m=(q[59, :3] - a["initial_q"][body, :3]).tolist(),
        displacement_1s_to_5_5s_m=(q[-1, :3] - q[59, :3]).tolist(),
    )
    results = [capture(Path("/tmp/colibri_joint_pose" + str(frame) + ".npz")) for frame in (1, 30, 120)]
    out = dict(
        motion=motion,
        frozen_capacity=results,
        scope="Capacity uses OLD canonical final-relax pose snapshots, not the new total-normal trajectory. Speculative impulses and all other joint/contact reactions stay fixed. Successful exact circular-cone certificate proves local static support feasible at that snapshot; failure of an inner polygon alone does not prove physical impossibility. No recorded phase data at the later startup displacement burst.",
    )
    Path("/tmp/colibri_startup_support_capacity.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
