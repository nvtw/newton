# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""CPU covariance audit of frozen physical response; not a live solver test."""

import json
from pathlib import Path

import numpy as np


def response(W, Jb, Js, Ds, old, reference, C, impulse, u):
    joint = np.concatenate((Jb, Js))
    diagonal = np.concatenate((np.zeros(len(Jb)), Ds))
    free = u + W @ C.T @ impulse
    target = np.concatenate((np.zeros(len(Jb)), Ds * old - reference))
    reaction = np.linalg.solve(joint @ W @ joint.T + np.diag(diagonal), -(joint @ free + target))
    return free + W @ joint.T @ reaction, reaction


def main():
    d = np.load("/tmp/colibri_coupled_support_reference.npz")
    metadata = json.loads(Path("/tmp/colibri_coupled_support_reference.json").read_text())
    snapshot = np.load("/tmp/colibri_support_relax330.npz")
    positions = snapshot["position"][metadata["bodies"]].astype(float)
    selected = np.concatenate([np.arange(3 * i, 3 * i + 3) for i in d["active"]])
    C = d["J"][selected]
    impulse = d["solution"] - d["initial"]
    W, Jb, Js, u = (d[k] for k in ("W", "Jb", "Js", "u"))
    v0, reaction0 = response(W, Jb, Js, d["Ds"], d["old_soft"], d["reference_soft"], C, impulse, u)
    np.testing.assert_allclose(v0, d["v"], atol=1e-11, rtol=1e-10)
    energy0 = float(0.5 * (v0 @ np.linalg.solve(W, v0) - u @ np.linalg.solve(W, u)))
    rng = np.random.default_rng(631)
    cases = []
    for index in range(12):
        rotation, _ = np.linalg.qr(rng.normal(size=(3, 3)))
        if np.linalg.det(rotation) < 0:
            rotation[:, 0] *= -1
        order = [0, 1] if index % 2 == 0 else [1, 0]
        transform = np.zeros_like(W)
        for new, old in enumerate(order):
            for offset in (0, 3):
                transform[6 * new + offset : 6 * new + offset + 3, 6 * old + offset : 6 * old + offset + 3] = rotation
        translation = rng.normal(size=3) * 10
        position = positions[order] @ rotation.T + translation
        wt = transform @ W @ transform.T
        ct = C @ transform.T
        ut = transform @ u
        vt, reaction = response(
            wt, Jb @ transform.T, Js @ transform.T, d["Ds"], d["old_soft"], d["reference_soft"], ct, impulse, ut
        )
        momentum = np.linalg.solve(wt, vt - ut).reshape(-1, 6)
        external = (ct.T @ impulse).reshape(-1, 6)
        error = momentum - external
        total = np.concatenate(
            (error[:, :3].sum(axis=0), (error[:, 3:] + np.cross(position, error[:, :3])).sum(axis=0))
        )
        energy = float(0.5 * (vt @ np.linalg.solve(wt, vt) - ut @ np.linalg.solve(wt, ut)))
        case = {
            "index": index,
            "swapped": bool(index % 2),
            "velocity_error": float(np.max(np.abs(vt - transform @ v0))),
            "reaction_error": float(np.max(np.abs(reaction - reaction0))),
            "energy_error_J": abs(energy - energy0),
            "momentum_minus_static_reaction": total.tolist(),
        }
        assert case["velocity_error"] < 1e-10, case
        assert case["reaction_error"] < 1e-10, case
        assert case["energy_error_J"] < 1e-12, case
        assert np.max(np.abs(total)) < 1e-9, case
        cases.append(case)
    report = {
        "passed": True,
        "scope": "Frozen physical response at accepted contact impulses; no contact re-solve or live scheduling claim",
        "cases": cases,
    }
    Path("/tmp/colibri_support_response_covariance.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
