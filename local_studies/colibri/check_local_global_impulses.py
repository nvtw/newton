"""Illustrative bilateral local/global impulse solve; not PhoenX or Coulomb."""

import json
from pathlib import Path

import numpy as np


def main():
    """Compare twelve-color local sweeps and momentum-preserving Schur corrections."""
    n = 25
    masses = np.ones(n)
    masses[n // 2] = 400.0
    centers = np.column_stack((np.arange(n) * 0.4, np.zeros(n), np.zeros(n)))
    inertia = masses[:, None] * np.array([0.12, 0.19, 0.23])
    mass = np.concatenate((np.repeat(masses[:, None], 3, axis=1), inertia), axis=1).ravel()
    inverse = 1.0 / mass
    jacobian = np.zeros((3 * (n - 1), 6 * n))
    for edge in range(n - 1):
        pivot = (centers[edge] + centers[edge + 1]) * 0.5 + [0.0, 0.15, -0.07]
        for axis, direction in enumerate(np.eye(3)):
            for body, sign in ((edge, -1.0), (edge + 1, 1.0)):
                jacobian[edge * 3 + axis, body * 6 : body * 6 + 6] = (
                    sign * np.r_[direction, np.cross(pivot - centers[body], direction)]
                )
    operator = (jacobian * inverse) @ jacobian.T
    colors = [[edge for edge in range(n - 1) if edge % 12 == color] for color in range(12)]
    for edges in colors:
        endpoints = [b for edge in edges for b in (edge, edge + 1)]
        assert len(endpoints) == len(set(endpoints))
    rng = np.random.default_rng(421)
    initial = rng.normal(size=6 * n) * 0.2
    # Coarse columns span long-wavelength multiplier fields, not an oracle root.
    basis = np.zeros((3 * (n - 1), 6))
    for edge in range(n - 1):
        for axis in range(3):
            basis[3 * edge + axis, axis] = 1.0
            basis[3 * edge + axis, axis + 3] = (edge - (n - 2) / 2) / (n - 1)
    basis, _ = np.linalg.qr(basis)
    coarse = basis.T @ operator @ basis

    def totals(v):
        momentum = (mass * v).reshape(n, 6)
        return np.r_[momentum[:, :3].sum(axis=0), (momentum[:, 3:] + np.cross(centers, momentum[:, :3])).sum(axis=0)]

    initial_momentum = totals(initial)
    initial_energy = 0.5 * np.sum(mass * initial**2)
    output = []
    for method in ("local_gs", "local_plus_coarse", "local_plus_exact"):
        velocity = initial.copy()
        accumulated = np.zeros(len(operator))
        history = []
        scalar_updates = 0
        global_solves = 0
        for cycle in range(9):
            residual = jacobian @ velocity
            reconstruction = initial + inverse * (jacobian.T @ accumulated)
            np.testing.assert_allclose(velocity, reconstruction, atol=2e-13, rtol=0)
            defect = totals(velocity) - initial_momentum
            np.testing.assert_allclose(defect, 0.0, atol=2e-11, rtol=0)
            impulse = jacobian.T @ accumulated
            work = float(impulse @ ((initial + velocity) * 0.5))
            energy = float(0.5 * np.sum(mass * velocity**2))
            assert abs(energy - initial_energy - work) < 2e-12
            if history:
                assert energy <= history[-1]["kinetic_energy_J"] + 2e-12
            history.append(
                {
                    "cycle": cycle,
                    "residual_l2_m_s": float(np.linalg.norm(residual)),
                    "residual_max_m_s": float(np.max(np.abs(residual))),
                    "kinetic_energy_J": energy,
                    "momentum_defect_Ns_Nms": defect.tolist(),
                    "work_balance_error_J": energy - initial_energy - work,
                    "scalar_row_updates": scalar_updates,
                    "global_solves": global_solves,
                }
            )
            if cycle == 8:
                break
            for edges in colors:
                # Disjoint edges commute inside each color; three joint axes use GS.
                for edge in edges:
                    for axis in range(3):
                        row = 3 * edge + axis
                        delta = -(jacobian[row] @ velocity) / operator[row, row]
                        velocity += inverse * jacobian[row] * delta
                        accumulated[row] += delta
                        scalar_updates += 1
            residual = jacobian @ velocity
            if method == "local_plus_exact":
                delta = np.linalg.solve(operator, -residual)
            elif method == "local_plus_coarse":
                delta = basis @ np.linalg.solve(coarse, -(basis.T @ residual))
            else:
                continue
            before = velocity.copy()
            velocity += inverse * (jacobian.T @ delta)
            accumulated += delta
            global_solves += 1
            correction_work = float((jacobian.T @ delta) @ ((before + velocity) * 0.5))
            assert correction_work <= 2e-12
        output.append({"method": method, "history": history})
    assert output[2]["history"][1]["residual_max_m_s"] < 1e-12
    report = {
        "scope": "Original fixed-pose bilateral illustration, not VBD/JGS2 implementation or Coulomb proof",
        "bodies": n,
        "mass_ratio": 400,
        "rows": len(operator),
        "physical_dofs": 6 * n,
        "colors": colors,
        "operator_condition": float(np.linalg.cond(operator)),
        "coarse_dimension": 6,
        "coarse_condition": float(np.linalg.cond(coarse)),
        "global_update": "delta_v=M^-1 J^T delta_lambda; exact A solve or six-dimensional Galerkin correction",
        "work_count": "Every cycle:72 scalar GS rows; optional one6x6 or72x72 solve. Factorization/setup excluded from counts; no speed claim.",
        "results": output,
    }
    Path("/tmp/local_global_impulses.json").write_text(json.dumps(report, indent=2))
    print(json.dumps({r["method"]: [h["residual_l2_m_s"] for h in r["history"]] for r in output}, indent=2))


if __name__ == "__main__":
    main()
