"""CPU-only connected normal block followed by native-law metric tangents.

Accept only unbiased frozen relaxation operators. Each owned normal row is
updated once per sweep; no scalar normal update is applied after the block.
This is a convergence diagnostic, not a live eight-substep trajectory.
"""

import json
from pathlib import Path

import numpy as np

from local_studies.colibri.reference_high_mass_connected_normals import active_set, tangent_step


def main(operator=None, snapshot=None, output="/tmp/high_mass_normal_sweeps_native_relax"):
    prefix = "/tmp/high_mass_mode_continuation_"
    op = np.load(operator or prefix + "live_245_1.npz")
    raw = np.load(snapshot or prefix + "live_before_245_1.npz")
    A, J, W = (op[k].astype(float) for k in ("A", "J", "inverse_mass"))
    initial, rhs = op["initial"].astype(float), op["rhs"].astype(float)
    points = op["selected_points"]
    assert np.all(op["normal_regularization"] == 0)
    assert np.all(raw["derived"][3, points] <= 0)
    assert np.all(raw["derived"][8, points] == 0), "PD rows need a separate operator"
    assert len(points) <= 12, "Exponential active-set oracle is deliberately bounded"
    u = np.concatenate((raw["velocity"], raw["angular_velocity"]), axis=1).astype(float).ravel()
    np.testing.assert_allclose(rhs, J @ u - A @ initial, atol=1e-12, rtol=0)
    h = raw["headers"].view(np.int32)
    mu_by_point = np.zeros(raw["impulses"].shape[1])
    ownership = np.zeros_like(mu_by_point, dtype=int)
    for c in range(int(raw["column_count"][0])):
        assert raw["headers"][3, c] == raw["headers"][4, c]
        ids = slice(h[5, c], h[5, c] + h[6, c])
        mu_by_point[ids] = raw["headers"][4, c]
        ownership[ids] += 1
    assert np.all(ownership[points] == 1)
    mu = mu_by_point[points]
    normal = np.arange(0, len(initial), 3)
    tangent = np.setdiff1d(np.arange(len(initial)), normal)
    B = A[np.ix_(normal, normal)]
    mass = np.linalg.pinv(W, hermitian=True)
    moving = raw["inverse_mass"] > 0
    result = {}
    saved = {}
    for method in ("scalar", "connected_normal"):
        value = initial.copy()
        history = []
        for sweep in range(8):
            if method == "connected_normal":
                value[normal] = active_set(B, rhs[normal] + A[np.ix_(normal, tangent)] @ value[tangent])
            for k in range(len(points)):
                n = 3 * k
                if method == "scalar":
                    value[n] = max(0, value[n] - (A[n] @ value + rhs[n]) / A[n, n])
                ids = np.array([n + 1, n + 2])
                K = A[np.ix_(ids, ids)]
                value[ids] = tangent_step(K, A[ids] @ value + rhs[ids], value[ids], mu[k] * value[n])
            gradient = A @ value + rhs
            nr = np.max(np.abs(np.minimum(value[normal] * np.diag(B), gradient[normal])))
            tr = []
            for k in range(len(points)):
                ids = np.array([3 * k + 1, 3 * k + 2])
                K = A[np.ix_(ids, ids)]
                target = tangent_step(K, gradient[ids], value[ids], mu[k] * value[3 * k])
                tr.append(np.linalg.norm(K @ (target - value[ids])))
            delta = J.T @ (value - initial)
            v = u + W @ delta
            energy = float((v @ mass @ v - u @ mass @ u) * 0.5)
            work = float(delta @ ((u + v) * 0.5))
            reaction = delta.reshape(-1, 6).copy()
            reaction[moving] = (mass @ (v - u)).reshape(-1, 6)[moving]
            linear = reaction[:, :3].sum(0)
            angular = (reaction[:, 3:] + np.cross(raw["positions"], reaction[:, :3])).sum(0)
            authored = delta.reshape(-1, 6)
            lever_defect = (authored[:, 3:] + np.cross(raw["positions"], authored[:, :3])).sum(0)
            history.append(
                {
                    "sweeps": sweep + 1,
                    "normal_residual_m_s": float(nr),
                    "tangent_residual_m_s": float(max(tr)),
                    "energy_change_J": energy,
                    "work_error_J": energy - work,
                    "linear_balance": linear.tolist(),
                    "angular_balance": angular.tolist(),
                    "stored_lever_angular_defect": lever_defect.tolist(),
                    "angular_accounting_error": (angular - lever_defect).tolist(),
                }
            )
            assert abs(energy - work) < 1e-10
            assert np.max(np.abs(angular - lever_defect)) < 1e-10
        result[method] = history
        saved[method] = value
    Path(output + ".json").write_text(json.dumps(result, indent=2))
    np.savez(output + ".npz", **saved)
    print(json.dumps({k: v[-1] for k, v in result.items()}, indent=2))


if __name__ == "__main__":
    main()
