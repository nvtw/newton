"""Local single-block joint-conditioned contact response and fixed-face check.

This is a linear reference, not a production Coulomb outer solver.
"""

import json
from pathlib import Path
import numpy as np
import warp as wp
from newton._src.solvers.phoenx.articulations.fixed_pattern_llt_queue import _block_sync
from local_studies.colibri.coulomb_semismooth import natural_map_evaluator


@wp.kernel(enable_backward=False)
def coupled_response(
    k: wp.array2d[wp.float64],
    rhs: wp.array2d[wp.float64],
    wct: wp.array2d[wp.float64],
    wbt: wp.array2d[wp.float64],
    free: wp.array[wp.float64],
    c: wp.array2d[wp.float64],
    face: wp.array[wp.float64],
    active: int,
    rows: int,
    factor: wp.array2d[wp.float64],
    solved: wp.array2d[wp.float64],
    response: wp.array2d[wp.float64],
    baseline: wp.array[wp.float64],
    impulse: wp.array[wp.float64],
    velocity: wp.array[wp.float64],
):
    lane = wp.tid()
    if lane < 36:
        i = lane // 6
        j = lane % 6
        factor[i, j] = k[i, j]
    _block_sync()
    for pivot in range(6):
        if lane == 0:
            factor[pivot, pivot] = wp.sqrt(factor[pivot, pivot])
        _block_sync()
        if lane > pivot and lane < 6:
            factor[lane, pivot] /= factor[pivot, pivot]
        _block_sync()
        if lane < 36:
            i = lane // 6
            j = lane % 6
            if i > pivot and j > pivot and i >= j:
                factor[i, j] -= factor[i, pivot] * factor[j, pivot]
        _block_sync()
    for col in range(lane, rows + 1, 128):
        for i in range(6):
            value = rhs[i, col]
            for j in range(i):
                value -= factor[i, j] * solved[j, col]
            solved[i, col] = value / factor[i, i]
        for back in range(6):
            i = 5 - back
            value = solved[i, col]
            for j in range(i + 1, 6):
                value -= factor[j, i] * solved[j, col]
            solved[i, col] = value / factor[i, i]
    _block_sync()
    for index in range(lane, 12 * (rows + 1), 128):
        bodyrow = index // (rows + 1)
        col = index % (rows + 1)
        value = wp.float64(0)
        for j in range(6):
            value += wbt[bodyrow, j] * solved[j, col]
        if col == 0:
            baseline[bodyrow] = free[bodyrow] + value
        else:
            response[bodyrow, col - 1] = wct[bodyrow, col - 1] - value
    _block_sync()
    if active >= 0:
        if lane == 0:
            normal_free = wp.float64(0)
            denominator = wp.float64(0)
            for i in range(12):
                normal_free += c[3 * active, i] * baseline[i]
                response_face = wp.float64(0)
                for j in range(rows):
                    response_face += response[i, j] * face[j]
                denominator += c[3 * active, i] * response_face
            impulse[rows] = -normal_free / denominator
        _block_sync()
        for row in range(lane, rows, 128):
            impulse[row] = face[row] * impulse[rows]
        _block_sync()
    if lane < 12:
        value = baseline[lane]
        for row in range(rows):
            value += response[lane, row] * impulse[row]
        velocity[lane] = value


def main():
    records = []
    for phase in ("biased", "relax"):
        d = np.load("/tmp/colibri_two_body_full_" + phase + ".npz")
        W, B, C = d["W"], d["B"], d["C"]
        old = d["initial"]
        free = d["velocity"] - W @ (C.T @ old + B.T @ d["old_joint"])
        K = B @ W @ B.T + np.diag(d["diagonal"])
        rhs = np.column_stack([d["targets"] - B @ free, B @ W @ C.T])
        rows = len(old)
        face = np.zeros(rows)
        active = -1
        if phase == "relax":
            lam = d["solution"].reshape(-1, 3)
            active = int(np.argmax(lam[:, 0]))
            # Outer reference certifies all other rows inactive. No rank cutoff.
            # Their minuscule numerical residual impulses are audited below.
            face[3 * active : 3 * active + 3] = lam[active] / lam[active, 0]

        def arr(x):
            return wp.array(np.asarray(x, dtype=np.float64), dtype=wp.float64, device="cuda:0")

        factor = wp.zeros((6, 6), dtype=wp.float64, device="cuda:0")
        solved = wp.zeros(rhs.shape, dtype=wp.float64, device="cuda:0")
        response = wp.zeros((12, rows), dtype=wp.float64, device="cuda:0")
        baseline = wp.zeros(12, dtype=wp.float64, device="cuda:0")
        impulse = wp.zeros(rows + 1, dtype=wp.float64, device="cuda:0")
        velocity = wp.zeros(12, dtype=wp.float64, device="cuda:0")
        wp.launch(
            coupled_response,
            dim=128,
            block_dim=128,
            inputs=[
                arr(K),
                arr(rhs),
                arr(W @ C.T),
                arr(W @ B.T),
                arr(free),
                arr(C),
                arr(face),
                active,
                rows,
                factor,
                solved,
                response,
                baseline,
                impulse,
                velocity,
            ],
            device="cuda:0",
        )
        Y = solved.numpy()
        G = response.numpy()
        vbar = baseline.numpy()
        expected = np.linalg.solve(K, rhs)
        P = W - W @ B.T @ np.linalg.solve(K, B @ W)
        physical_error = float(np.max(np.abs(G - P @ C.T)))
        linear_error = float(np.max(np.abs(K @ Y - rhs)))
        assert linear_error < 1e-8 and physical_error < 1e-8
        # No contact-space factorization or rank truncation: rank bound follows
        # five independent exact hard rows of B on twelve physical coordinates.
        hard = B[d["diagonal"] == 0]
        sv = np.linalg.svd(hard, compute_uv=False)
        assert sv[-1] > 0.1
        report = dict(
            phase=phase,
            points=rows // 3,
            contact_rows=rows,
            body_dofs=12,
            joint_rows=6,
            independent_hard_joint_singular_values=sv.tolist(),
            constrained_physical_rank_bound=7,
            normal_compliant_rows=int(np.count_nonzero(d["regularization"])),
            joint_factor_eigenvalues=np.linalg.eigvalsh(K).tolist(),
            gpu_joint_equation_max=linear_error,
            gpu_physical_response_max_error=physical_error,
            cpu_solve_difference=float(np.max(np.abs(Y - expected))),
            contact_mobility_singular_values=np.linalg.svd(d["A"], compute_uv=False)[:12].tolist(),
        )
        if active >= 0:
            lam = impulse.numpy()[:rows]
            v = velocity.numpy()
            joint = Y[:, 0] - Y[:, 1:] @ lam
            evaluate, *_ = natural_map_evaluator(d["A"], d["rhs"], d["regularization"], d["mu"])
            residual = float(np.max(np.abs(evaluate(lam)[0])))
            constitutive = float(np.max(np.abs(B @ v + d["diagonal"] * joint - d["targets"])))
            independent_v = free + W @ (C.T @ lam + B.T @ joint)
            dv = v - d["velocity"]
            dp = np.linalg.solve(W, dv)
            work = float(dp @ ((v + d["velocity"]) * 0.5))
            energy = float(0.5 * (v @ np.linalg.solve(W, v) - d["velocity"] @ np.linalg.solve(W, d["velocity"])))
            report.update(
                active_point=active,
                all_contacts_natural_residual=residual,
                joint_residual=constitutive,
                physical_impulse_response_error=float(np.max(np.abs(v - independent_v))),
                accepted_cpu_velocity_difference=float(np.max(np.abs(v - d["updated"]))),
                energy_change=energy,
                midpoint_work=work,
                work_error=abs(work - energy),
            )
            assert residual < 1e-8 and constitutive < 1e-8
            assert np.max(np.abs(v - d["updated"])) < 1e-8
        else:
            evaluate, *_ = natural_map_evaluator(d["A"], d["rhs"], d["regularization"], d["mu"])
            report["existing_outer_residual"] = float(np.max(np.abs(evaluate(d["solution"])[0])))
            report["full_coulomb_accepted"] = False
        np.savez(
            "/tmp/single_block_support_" + phase + ".npz",
            G=G,
            baseline=vbar,
            solved=Y,
            impulse=impulse.numpy()[:rows],
            velocity=velocity.numpy(),
            K=K,
            rhs=rhs,
        )
        records.append(report)
    Path("/tmp/single_block_support_reference.json").write_text(json.dumps(records, indent=2))
    print(json.dumps(records, indent=2))


if __name__ == "__main__":
    main()
