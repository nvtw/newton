"""Tree-condensed physical joint mobility using the unchanged reformed rows."""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import sympy as sp


def main():
    import scipy.linalg as la

    parser = argparse.ArgumentParser()
    parser.add_argument("--biased", action="store_true")
    parser.add_argument("--operator", default="/tmp/colibri_joint_reformed_input.npz")
    parser.add_argument("--snapshot", default="/tmp/colibri_support_relax330.npz")
    parser.add_argument("--reference-mobility", default="/tmp/colibri_joint_constrained_mobility.npz")
    parser.add_argument("--reference-response", default="/tmp/colibri_joint_constrained.npz")
    parser.add_argument("--topology-from")
    parser.add_argument("--output", default="/tmp/colibri_joint_tree")
    args = parser.parse_args()
    p = np.load(args.operator)
    s = np.load(args.snapshot)
    ref = np.load(args.reference_mobility)
    base_ref = np.load(args.reference_response)
    active = ref["active"]
    nb = len(active)
    J = p["J"][:, active, :].reshape(188, -1)
    R = p["compliance"]
    drive = np.flatnonzero(R > 0)
    bias = np.zeros(188)
    if args.biased:
        for joint, count in enumerate(s["joint_row_count"]):
            structural = s["joint_structural_index"][joint]
            for row in s["joint_row_indices"][joint, :count]:
                bias[row] = s["joint_bias"][structural, s["joint_row_local"][row]]
    parent = list(range(nb))

    def find(x):
        while parent[x] != x:
            x = parent[x]
        return x

    edges = []
    loops = []
    tree_rows = []
    for joint, count in enumerate(s["joint_row_count"]):
        rows = [int(r) for r in s["joint_row_indices"][joint, :count] if R[r] == 0]
        endpoints = np.flatnonzero(np.any(p["J"][rows] != 0, axis=(0, 2)))
        assert len(endpoints) == 2 and all(b in active for b in endpoints)
        a, b = [int(np.flatnonzero(active == x)[0]) for x in endpoints]
        if find(a) != find(b):
            parent[find(a)] = find(b)
            edges.append((a, b, rows))
            tree_rows.extend(rows)
        else:
            loops.extend(rows)
    adjacency = [[] for _ in active]
    for a, b, rows in edges:
        adjacency[a].append((b, rows))
        adjacency[b].append((a, rows))
    oriented = []
    dimension = 6 * nb - len(tree_rows)
    N = np.zeros((nb, 6, dimension))
    particular = np.zeros((nb, 6))
    visited = set()
    cursor = 0
    roots = []
    for root in range(nb):
        if root in visited:
            continue
        roots.append(root)
        N[root, :, cursor : cursor + 6] = np.eye(6)
        cursor += 6
        visited.add(root)
        queue = [root]
        while queue:
            a = queue.pop(0)
            for b, rows in adjacency[a]:
                if b in visited:
                    continue
                visited.add(b)
                queue.append(b)
                A = J[np.ix_(rows, range(6 * b, 6 * b + 6))]
                B = J[np.ix_(rows, range(6 * a, 6 * a + 6))]
                q, t = la.qr(A.T, mode="full")
                m = len(rows)
                oriented.append((a, b, rows, q, t))
                transfer = -q[:, :m] @ la.solve_triangular(t[:m, :m].T, B, lower=True)
                particular[b] = transfer @ particular[a] - q[:, :m] @ la.solve_triangular(
                    t[:m, :m].T, bias[rows], lower=True
                )
                N[b] = transfer @ N[a]
                N[b, :, cursor : cursor + 6 - m] = q[:, m:]
                cursor += 6 - m
    assert cursor == dimension
    N = N.reshape(nb * 6, dimension)
    M = la.block_diag(*(np.linalg.inv(p["W"][b]) for b in active))
    reduced_mass = N.T @ M @ N
    chol = la.cholesky(reduced_mass, lower=True)
    # S S^T is the inverse reduced physical mass.
    S = la.solve_triangular(chol.T, np.eye(dimension), lower=False)
    Tbody = N @ S
    remaining = loops + list(drive)
    order = tree_rows + remaining
    assert len(order) == 188 and len(set(order)) == 188
    F = np.concatenate((J.T, np.eye(188)[R > 0]), axis=0)
    exact = sp.polys.matrices.DomainMatrix.from_Matrix(
        sp.Matrix([[sp.Rational(float(x)) for x in row] for row in F[:, order]])
    )
    selected = [order[k] for k in exact.rref()[1]]
    assert set(tree_rows).issubset(selected)
    loopkeep = [r for r in selected if r not in tree_rows]
    if args.topology_from:
        topology = np.load(args.topology_from)
        assert np.array_equal(tree_rows, topology["tree_rows"])
        loopkeep = topology["loop_rows"].tolist()
        indices = [order.index(row) for row in tree_rows + loopkeep]
        assert len(exact.extract(range(exact.shape[0]), indices).rref()[1]) == len(indices) == 184
    K = np.column_stack((J[loopkeep] @ Tbody, np.diag(np.sqrt(R))[np.ix_(loopkeep, drive)]))
    scale = 1 / np.linalg.norm(K, axis=1)
    start = time.perf_counter()
    Q, U = la.qr((K * scale[:, None]).T, mode="full")
    elapsed = time.perf_counter() - start
    rank = len(loopkeep)
    G = Tbody @ Q[:dimension, rank:]
    slack = Q[dimension:, rank:]
    # First project initial velocity onto tree-hard constraints in the physical metric.
    initial = p["initial"][active].ravel()
    particular = particular.ravel()
    tree_initial = particular + Tbody @ (Tbody.T @ M @ (initial - particular))
    ref_joint = s["joint_reference"] - R * s["after_joint_accumulated"] - bias
    residual = J @ tree_initial - ref_joint
    a = la.solve_triangular(U[:rank, :rank].T, -residual[loopkeep] * scale, lower=True)
    z = Q[:, :rank] @ a
    after = tree_initial + Tbody @ z[:dimension]
    lamloop = np.zeros(188)
    lamloop[loopkeep] = scale * la.solve_triangular(U[:rank, :rank], a)
    allres = J @ after + R * (s["after_joint_accumulated"] + lamloop) - s["joint_reference"] + bias
    total_delta = lamloop.copy()
    remaining_force = (M @ (after - initial) - J.T @ lamloop).reshape(nb, 6)
    for a, b, rows, q, t in reversed(oriented):
        count = len(rows)
        values = la.solve_triangular(t[:count, :count], q[:, :count].T @ remaining_force[b])
        total_delta[rows] = values
        for body in (a, b):
            remaining_force[body] -= J[np.ix_(rows, range(6 * body, 6 * body + 6))].T @ values
    physical = (M @ (after - initial)).reshape(nb, 6)
    position = s["position"][active].astype(float)
    momentum = physical[:, :3].sum(axis=0)
    angular = np.sum(physical[:, 3:] + np.cross(position, physical[:, :3]), axis=0)
    energy = (after @ M @ after - initial @ M @ initial) * 0.5
    work = total_delta @ (J @ ((after + initial) * 0.5))
    header = s["headers"].view(np.int32)
    contact = []
    for col in range(int(s["column_count"][0])):
        x, y = header[1:3, col]
        for point in range(header[5, col], header[5, col] + header[6, col]):
            if s["derived"][3, point] > 0:
                continue
            normal = s["lambdas"][:3, point].astype(float)
            tangent = s["lambdas"][3:6, point].astype(float)
            axes = np.array([normal, tangent, np.cross(normal, tangent)])
            row = np.zeros((3, len(s["position"]), 6))
            row[:, x] = -np.concatenate((axes, np.cross(s["derived"][9:12, point].astype(float), axes)), axis=1)
            row[:, y] = np.concatenate((axes, np.cross(s["derived"][12:15, point].astype(float), axes)), axis=1)
            contact.extend(row[:, active].reshape(3, -1))
    c = np.array(contact)
    h = (c @ G) @ (c @ G).T
    hr = (c @ ref["G"]) @ (c @ ref["G"]).T
    check = J @ G
    check[drive] += np.sqrt(R[drive, None]) * slack
    out = {
        "biased": args.biased,
        "max_bias": float(abs(bias).max()),
        "bodies": nb,
        "tree_edges": len(edges),
        "forest_roots": [int(active[b]) for b in roots],
        "tree_hard_rows": len(tree_rows),
        "tree_velocity_dofs": dimension,
        "remaining_rows_including_drives": len(remaining),
        "independent_remaining_rows": rank,
        "reduced_QR_shape": list(K.T.shape),
        "complement_columns": G.shape[1],
        "tree_basis_residual": float(abs(J[tree_rows] @ N).max()),
        "full_joint_complement_residual": float(abs(check).max()),
        "all_original_equation_residual": float(abs(allres).max()),
        "baseline_dv_difference_to_reference": None
        if args.biased
        else float(abs(after - initial - base_ref["dv"].ravel()).max()),
        "contact_mobility_relative_error": float(np.linalg.norm(h - hr) / np.linalg.norm(hr)),
        "contact_mobility_max_error": float(abs(h - hr).max()),
        "small_QR_CPU_seconds": elapsed,
    }
    null = exact.nullspace().to_Matrix()
    null_rhs = []
    residual_initial = J @ initial + R * s["after_joint_accumulated"] - s["joint_reference"] + bias
    for vector in null.tolist():
        maximum = max(abs(x) for x in vector)
        null_rhs.append(
            float(sum(x * sp.Rational(float(residual_initial[order[k]])) for k, x in enumerate(vector)) / maximum)
        )
    out["exact_null_rhs_errors"] = null_rhs
    out["tree_affine_residual"] = float(abs(J[tree_rows] @ particular + bias[tree_rows]).max())
    out["linear_equation_accepted"] = float(abs(allres).max()) < 1e-8
    out["reference_response"] = args.reference_response
    out["reference_mobility"] = args.reference_mobility
    out.update(
        tree_reaction_force_residual=float(abs(remaining_force).max()),
        deltaP=momentum.tolist(),
        deltaL=angular.tolist(),
        deltaKE=float(energy),
        joint_midpoint_work=float(work),
        joint_work_error=float(abs(energy - work)),
    )
    response = G @ (c @ G).T
    drive_response = (slack @ (c @ G).T) / np.sqrt(R[drive])[:, None]
    response_work = np.sum(c.T * response, axis=0)
    kinetic = np.sum(response * (M @ response), axis=0)
    penalty = np.sum(R[drive, None] * drive_response**2, axis=0)
    out["contact_relative_virtual_work_error"] = float(
        np.max(abs(response_work - kinetic - penalty) / np.maximum(abs(response_work), 1))
    )
    np.savez(args.output + "_response.npz", delta=total_delta, dv=(after - initial).reshape(nb, 6), active=active)
    np.savez(
        args.output + "_mobility.npz",
        qr_input=(K * scale[:, None]).T,
        loop_velocity_rows=J[loopkeep] @ N,
        drive_sqrt_rows=np.diag(np.sqrt(R))[np.ix_(loopkeep, drive)],
        reduced_mass=reduced_mass,
        K=K,
        K_all=np.column_stack((J[remaining] @ Tbody, np.diag(np.sqrt(R))[np.ix_(remaining, drive)])),
        G=G,
        slack=slack,
        active=active,
        drive_rows=drive,
        R=R,
        N=N,
        tree_rows=tree_rows,
        loop_rows=loopkeep,
        Tbody=Tbody,
        after=after,
        delta_drive=lamloop,
    )
    Path(args.output + "_mobility.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
