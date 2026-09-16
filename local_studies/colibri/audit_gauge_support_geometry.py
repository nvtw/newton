"""Certify a frozen support-line null motion from exact physical primitives."""

import json
from pathlib import Path

import mpmath as mp
import numpy as np
import sympy as sp

from local_studies.colibri.bounded_island_coulomb import response_factor


def main():
    mp.mp.dps = 70
    raw = np.load("/tmp/high_mass_window120_live_before_362_5.npz")
    op = np.load("/tmp/high_mass_window120_live_362_5.npz")
    consistent = np.load("/tmp/high_mass_consistent960_362_5.npz")
    h = raw["headers"].view(np.int32)

    def rational(x):
        return sp.Rational(float(x))

    positions = [sp.Matrix([rational(x) for x in row]) for row in raw["positions"]]
    J = sp.zeros(len(op["selected_points"]) * 3, 18)
    pivots = []
    pairs = []
    for index, p in enumerate(op["selected_points"]):
        c = next(c for c in range(int(raw["column_count"][0])) if h[5, c] <= p < h[5, c] + h[6, c])
        a, b = map(int, h[1:3, c])
        ra = sp.Matrix([rational(x) for x in raw["derived"][9:12, p]])
        rb = sp.Matrix([rational(x) for x in raw["derived"][12:15, p]])
        pivot = (positions[a] + ra + positions[b] + rb) / 2
        pivots.append(pivot)
        pairs.append([a, b])
        for k in range(3):
            axis = sp.Matrix([rational(x) for x in op["J"][3 * index + k, 6 * b : 6 * b + 3]])
            for body, sign in ((a, -1), (b, 1)):
                wrench = axis.col_join((pivot - positions[body]).cross(axis))
                for j in range(6):
                    J[3 * index + k, 6 * body + j] = sign * wrench[j]
    ground = [p for p, pair in zip(pivots, pairs, strict=False) if pair[0] == 0]
    omega = ground[1] - ground[0]
    twist = sp.zeros(18, 1)
    for body in (1, 2):
        v = omega.cross(positions[body] - ground[0])
        for j in range(3):
            twist[6 * body + j] = v[j]
            twist[6 * body + 3 + j] = omega[j]
    assert J * twist == sp.zeros(J.rows, 1)
    dynamic = J[:, 6:18]
    gauges = dynamic.T.nullspace()
    N_exact = sp.Matrix.hstack(*gauges)
    assert dynamic.T * N_exact == sp.zeros(12, 10)
    N = np.linalg.qr(np.array(N_exact, dtype=float))[0]
    rank = dynamic.rank()
    assert rank == 11
    W = op["inverse_mass"]
    L = np.zeros((18, 12))
    for body in (1, 2):
        L[6 * body : 6 * body + 6, 6 * (body - 1) : 6 * body] = np.linalg.cholesky(
            W[6 * body : 6 * body + 6, 6 * body : 6 * body + 6]
        )
    Jmp = mp.matrix([[mp.mpf(str(J[i, j].p)) / mp.mpf(str(J[i, j].q)) for j in range(18)] for i in range(J.rows)])
    Fmp = Jmp * mp.matrix(L.tolist())
    s = mp.svd(Fmp, compute_uv=False)
    _, _, vh = np.linalg.svd(response_factor(consistent["J"], W), full_matrices=False)
    numerical_twist = L @ vh[-1]
    exact_twist = np.array(twist, dtype=float).ravel()
    cosine = abs(float(numerical_twist @ exact_twist / (np.linalg.norm(numerical_twist) * np.linalg.norm(exact_twist))))
    initial = consistent["solution"]
    velocity = (consistent["A"] @ initial + consistent["rhs"]).reshape(-1, 3)
    direction = velocity[0, 1:] / np.linalg.norm(velocity[0, 1:])
    E = sp.zeros(2, J.rows)
    for k in range(2):
        E[k, 0] = rational(0.5 * direction[k])
        E[k, k + 1] = 1
    EN = E * N_exact
    equality_rhs = -E * sp.Matrix([rational(x) for x in initial])
    left = EN.T.nullspace()[0]
    incompatibility = float(abs((left.T * equality_rhs)[0]) / left.norm())
    equality_certificate = {
        "exact_gauge_sliding_rank": EN.rank(),
        "exact_augmented_rank": EN.row_join(equality_rhs).rank(),
        "fixed_sliding_equality_incompatibility_Ns": incompatibility,
        "point0_impulse_gauge_rank": N_exact[:3, :].rank(),
        "scope": "Fixed original candidate velocity and resolved point0 sliding direction. Rank augmented exceeds rank, so no exact response-neutral impulse correction satisfies this sliding law.",
    }
    result = {
        "case": "362_5",
        "gauge_feasibility_certificate": equality_certificate,
        "pairs": pairs,
        "world_pivots": [list(map(float, p)) for p in pivots],
        "exact_rational_dynamic_rank": rank,
        "pure_dual_dimension": J.rows - rank,
        "exact_symbolic_null_residual": 0,
        "common_rotation_axis": list(map(float, omega / omega.norm())),
        "body_twists_unit_omega": (exact_twist / float(omega.norm())).reshape(3, 6).tolist(),
        "primitive_recomputed_singular_values_70digit": [float(x) for x in s],
        "rounded_smallest_mode_twist_cosine": cosine,
        "max_J_difference": float(np.max(np.abs(np.array(J, dtype=float) - consistent["J"]))),
        "scope": "Explicit exact-rational shared-pivot cross products; original directions and physical inverse mass retained. No mode truncation or live solve.",
    }
    Path("/tmp/high_mass_gauge_support_geometry.json").write_text(json.dumps(result, indent=2))
    np.savez(
        "/tmp/high_mass_gauge_support_geometry.npz",
        J_exact_rounded=np.array(J, dtype=float),
        twist=exact_twist,
        L=L,
        N=N,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
