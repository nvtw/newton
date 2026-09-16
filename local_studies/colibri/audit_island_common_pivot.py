"""Explicit changed-geometry rank diagnostic; never modifies a live operator."""

import json
from pathlib import Path

import mpmath as mp
import numpy as np


def cross(a, b):
    return [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]


def main():
    mp.mp.dps = 70
    raw = np.load("/tmp/high_mass_window120_live_before_245_1.npz")
    op = np.load("/tmp/high_mass_window120_live_245_1.npz")
    captured = np.load("/tmp/high_mass_seed_null_rejection.npz")
    h = raw["headers"].view(np.int32)
    selected = op["selected_points"]
    J0 = op["J"]
    W = op["inverse_mass"]
    nb = len(W) // 6
    moving = [b for b in range(nb) if W[6 * b, 6 * b] > 0]
    L = mp.zeros(nb * 6, len(moving) * 6)
    for index, b in enumerate(moving):
        local = np.linalg.cholesky(W[6 * b : 6 * b + 6, 6 * b : 6 * b + 6])
        for i in range(6):
            for j in range(6):
                L[6 * b + i, 6 * index + j] = mp.mpf(float(local[i, j]))
    D = mp.matrix(captured["local_derivative"].tolist())
    variants = {"original_stored_F": mp.matrix(captured["F"].tolist())}
    matrices = {}
    shifts = []
    for shared in (False, True):
        J = mp.zeros(*J0.shape)
        for index, p in enumerate(selected):
            col = next(c for c in range(int(raw["column_count"][0])) if h[5, c] <= p < h[5, c] + h[6, c])
            a, b = map(int, h[1:3, col])
            pa = [mp.mpf(float(v)) for v in raw["positions"][a]]
            pb = [mp.mpf(float(v)) for v in raw["positions"][b]]
            ra = [mp.mpf(float(v)) for v in raw["derived"][9:12, p]]
            rb = [mp.mpf(float(v)) for v in raw["derived"][12:15, p]]
            pivot = [(pa[i] + ra[i] + pb[i] + rb[i]) / 2 for i in range(3)]
            if shared:
                shifts.append(float(mp.sqrt(sum((pa[i] + ra[i] - pivot[i]) ** 2 for i in range(3)))))
                ra = [pivot[i] - pa[i] for i in range(3)]
                rb = [pivot[i] - pb[i] for i in range(3)]
            for k in range(3):
                # Preserve the exact original stored direction, including the
                # original t2 rounding; isolate lever/cross-product arithmetic.
                axis = [mp.mpf(float(v)) for v in J0[3 * index + k, 6 * b : 6 * b + 3]]
                wa = cross(ra, axis)
                wb = cross(rb, axis)
                for j in range(3):
                    J[3 * index + k, 6 * a + j] = -axis[j]
                    J[3 * index + k, 6 * b + j] = axis[j]
                    J[3 * index + k, 6 * a + 3 + j] = -wa[j]
                    J[3 * index + k, 6 * b + 3 + j] = wb[j]
        name = "shared_pivot_exact_cross" if shared else "original_levers_exact_cross"
        variants[name] = J * L
        matrices[name + "_J"] = np.array(J.tolist(), dtype=float)
    results = {}
    for name, F in variants.items():
        stacked = mp.matrix(F.cols + D.rows, F.rows)
        for i in range(F.cols):
            for j in range(F.rows):
                stacked[i, j] = F[j, i]
        for i in range(D.rows):
            for j in range(D.cols):
                stacked[F.cols + i, j] = D[i, j]
        singular = mp.svd(stacked, compute_uv=False)
        results[name] = {
            "singular_values_70digit": [float(v) for v in singular],
            "F_difference_inf": float(
                max(abs(F[i, j] - variants["original_stored_F"][i, j]) for i in range(F.rows) for j in range(F.cols))
            ),
        }
        matrices[name + "_F"] = np.array(F.tolist(), dtype=float)
    results["max_shared_pivot_shift_m"] = max(shifts)
    results["scope"] = (
        "Fixed original local derivative D; changed J/F explicitly, original axes and physical W retained. No solve or mode deletion."
    )
    Path("/tmp/high_mass_common_pivot_rank.json").write_text(json.dumps(results, indent=2))
    np.savez("/tmp/high_mass_common_pivot_rank.npz", **matrices)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
