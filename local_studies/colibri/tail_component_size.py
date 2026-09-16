"""Size and condition a one-ring tail component before any nonlinear solve."""

import json
from pathlib import Path

import numpy as np

from newton._src.solvers.phoenx.body import inertia_sym6_unpack_np
from newton._src.solvers.phoenx.constraints import constraint_joint as schema


def main(core=None, prefix="/tmp/colibri_tail_component"):
    d = np.load("/tmp/colibri_support_relax330.npz")
    labels = ["WORLD", *np.load("/tmp/colibri_support_relax_prefix330.npz")["labels"].tolist()]
    # Seed the dominant gear/rack connected subtree, then retain every incident
    # row and both reaction endpoints. Boundary endpoints are not recursively expanded.
    core = {25, 26, 27, 28, 29, 30, 33, 36} if core is None else set(core)
    jd = d["joint_data"].view(np.int32)
    h = d["headers"].view(np.int32)
    joints = []
    columns = []
    bodies = set(core)
    for j in range(int(d["num_joints"][0])):
        pair = jd[[int(schema._OFF_BODY1), int(schema._OFF_BODY2)], j]
        if core.intersection(pair):
            joints.append(j)
            bodies.update(pair)
    for col in range(int(d["column_count"][0])):
        pair = h[1:3, col]
        if core.intersection(pair):
            columns.append(col)
            bodies.update(pair)
    bodies = sorted(int(b) for b in bodies if d["inverse_mass"][b] > 0)
    slots = {body: i for i, body in enumerate(bodies)}
    nd = 6 * len(bodies)
    W = np.zeros((nd, nd))
    inv_i = inertia_sym6_unpack_np(d["inverse_inertia"]).astype(float)
    for body, i in slots.items():
        W[6 * i : 6 * i + 3, 6 * i : 6 * i + 3] = np.eye(3) * d["inverse_mass"][body]
        W[6 * i + 3 : 6 * i + 6, 6 * i + 3 : 6 * i + 6] = inv_i[body]
    hard = []
    soft = []
    joint_rows = []
    for j in joints:
        a, b = jd[[int(schema._OFF_BODY1), int(schema._OFF_BODY2)], j]
        structural = int(d["joint_structural_index"][j])
        for k in range(int(d["joint_row_count"][j])):
            row = int(d["joint_row_indices"][j, k])
            local = int(d["joint_row_local"][row])
            jac = np.zeros(nd)
            for body, key in ((a, "joint_wrench0"), (b, "joint_wrench1")):
                if body in slots:
                    jac[6 * slots[body] : 6 * slots[body] + 6] += d[key][structural, local]
            (soft if d["joint_row_dynamic"][row] else hard).append(jac)
            joint_rows.append(row)
    Jb = np.array(hard)
    L = np.linalg.cholesky(W)
    U, S, Vt = np.linalg.svd(Jb @ L, full_matrices=True)
    threshold = np.finfo(float).eps * max(Jb.shape) * S[0]
    rank = int(np.count_nonzero(S > threshold))
    R = L @ Vt[rank:].T
    points = []
    jacobians = []
    active = []
    for col in columns:
        a, b = h[1:3, col]
        for p in range(h[5, col], h[5, col] + h[6, col]):
            n = d["lambdas"][:3, p].astype(float)
            t = d["lambdas"][3:6, p].astype(float)
            axes = np.array([n, t, np.cross(n, t)])
            jac = np.zeros((3, nd))
            for body, sign, lever in ((a, -1, d["derived"][9:12, p]), (b, 1, d["derived"][12:15, p])):
                if body in slots:
                    jac[:, 6 * slots[body] : 6 * slots[body] + 6] += sign * np.concatenate(
                        (axes, np.cross(lever, axes)), axis=1
                    )
            points.append(int(p))
            jacobians.extend(jac)
            if d["derived"][3, p] <= 0:
                active.append(len(points) - 1)
    J = np.array(jacobians)
    selected = (3 * np.array(active)[:, None] + np.arange(3)).ravel()
    C = J[selected]
    A = (C @ R) @ (C @ R).T
    singular = np.linalg.svd(C @ R, compute_uv=False)
    cut = np.finfo(float).eps * max((C @ R).shape) * singular[0]
    crank = int(np.count_nonzero(singular > cut))
    external_joints = []
    external_cols = []
    for j in range(int(d["num_joints"][0])):
        if j not in joints and set(bodies).intersection(jd[[int(schema._OFF_BODY1), int(schema._OFF_BODY2)], j]):
            external_joints.append(j)
    for col in range(int(d["column_count"][0])):
        if col not in columns and set(bodies).intersection(h[1:3, col]):
            external_cols.append(col)
    report = {
        "core": [labels[b] for b in sorted(core)],
        "bodies": {b: labels[b] for b in bodies},
        "joint_ids": joints,
        "contact_columns": len(columns),
        "all_points": len(points),
        "free_points": len(active),
        "fixed_speculative_points": len(points) - len(active),
        "physical_dofs": nd,
        "hard_rows": len(hard),
        "soft_rows": len(soft),
        "bounded_rows": int(np.count_nonzero(d["joint_bounded_drive"][joint_rows])),
        "joint_rank": rank,
        "null_dofs": nd - rank,
        "joint_massweighted_condition": float(S[0] / S[rank - 1]),
        "contact_rank": crank,
        "contact_unknowns": 3 * len(active),
        "contact_nonzero_condition": float(singular[0] / singular[crank - 1]),
        "physical_W_condition": float(np.linalg.cond(W)),
        "min_body_mass_kg": float(min(1 / d["inverse_mass"][b] for b in bodies)),
        "max_body_mass_kg": float(max(1 / d["inverse_mass"][b] for b in bodies)),
        "external_joint_ids": external_joints,
        "external_contact_columns": external_cols,
    }
    Path(prefix + "_size.json").write_text(json.dumps(report, indent=2))
    np.savez_compressed(
        prefix + "_operator.npz",
        W=W,
        Jb=Jb,
        J=J,
        A=A,
        R=R,
        points=points,
        active=active,
        bodies=bodies,
        joints=joints,
        columns=columns,
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
