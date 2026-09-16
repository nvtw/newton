"""CPU tests of the isolated literal GPU patch-friction subset and its limits."""

import json
from pathlib import Path

import numpy as np

from local_studies.colibri.physx_patch_reference import F, Patch, correlate, prepare, refresh, solve, writeback


def main():
    positions = np.array([[0, 0, 0.1], [0, 0, -0.1]], dtype=F)
    rotations = np.array([np.eye(3), np.eye(3)], dtype=F)
    points = np.array([[-0.2, 0, 0], [0, 0, 0], [0.2, 0, 0]], dtype=F)
    gaps = np.zeros(3, dtype=F)
    normal = np.array([0, 0, 1], dtype=F)
    mass = np.array([1, 4], dtype=F)
    inv = F(1) / mass
    inertia = np.array([np.diag([0.2, 0.3, 0.4]), np.diag([0.8, 1.2, 1.6])], dtype=F)
    inv_i = np.linalg.inv(inertia).astype(F)
    patch = refresh(points, gaps, normal, positions, rotations, 0.01, 0.02)
    assert len(patch.local0) == 2
    np.testing.assert_array_equal(patch.local0[:, 0], np.array([-0.2, 0.2], dtype=F))
    assert correlate(patch, positions, rotations, normal, 3, 0.02)
    shifted = positions.copy()
    shifted[1, 0] += F(0.1)
    assert correlate(patch, shifted, rotations, normal, 3, 0.02), "Source correlation checks normal separation only"
    shifted[1, 2] += F(0.03)
    assert not correlate(patch, shifted, rotations, normal, 3, 0.02)
    patch.broken = True
    assert not correlate(patch, positions, rotations, normal, 3, 0.02)
    patch.broken = False
    zero = np.zeros((2, 3), dtype=F)
    records = []
    for separation in [0, 0.001]:
        current = Patch(patch.local0.copy(), patch.local1.copy(), patch.normal0.copy(), patch.normal1.copy())
        current.local1[:, 2] += F(separation)
        linear = zero.copy()
        linear[0, 0] = F(1)
        state = prepare(current, positions, rotations, normal, linear, inv, inv_i, 0.001, 1)
        np.testing.assert_allclose(state["p8"], 0.8, rtol=0, atol=2e-8)
        updated, angular, ledger = solve(state, linear, zero, inv, inv_i, F(0.3) + F(0.7), 0.5, 0.4, zero, zero)
        assert all(np.linalg.norm(row) <= F(0.2) + F(1e-7) for row in state["impulse"])
        assert state["broken"]
        writeback(current, state)
        assert current.broken
        dp = ledger[:, :3].astype(float).sum(axis=0)
        dl = (ledger[:, 3:].astype(float) + np.cross(positions.astype(float), ledger[:, :3].astype(float))).sum(axis=0)
        energy = 0.5 * np.sum(mass[:, None].astype(float) * (updated.astype(float) ** 2 - linear.astype(float) ** 2))
        energy += 0.5 * sum(
            w.astype(float) @ i.astype(float) @ w.astype(float) for w, i in zip(angular, inertia, strict=True)
        )
        midpoint = np.c_[(updated.astype(float) + linear) / 2, angular.astype(float) / 2]
        work = float(np.sum(ledger.astype(float) * midpoint))
        assert np.max(abs(dp)) < 1e-7 and abs(energy - work) < 2e-7
        if separation == 0:
            assert np.max(abs(dl)) < 1e-7
        else:
            assert np.max(abs(dl)) > 1e-5, "Independent separated anchors must expose angular defect"
        records.append(
            {
                "anchor_separation_m": separation,
                "impulses": state["impulse"].tolist(),
                "linear_momentum_residual": dp.tolist(),
                "angular_momentum_change": dl.tolist(),
                "delta_ke": energy,
                "impulse_work": work,
                "work_error": energy - work,
            }
        )
    # Latest-solve overwrite: a new trial can clear header broken, while previous
    # writeback patch remains broken until refresh creates a fresh patch.
    state = prepare(patch, positions, rotations, normal, zero, inv, inv_i, 0.001, 1)
    state["impulse"][:] = np.array([0, 0.2475], dtype=F)
    solve(state, zero, zero, inv, inv_i, 1, 0.5, 0.5, zero, zero)
    assert not state["broken"]
    state["broken"] = True
    state["impulse"][:] = 0
    solve(state, zero, zero, inv, inv_i, 1, 0.5, 0.5, zero, zero)
    assert not state["broken"]
    # Accumulated motion, independently of instantaneous velocity, enters target.
    delta = zero.copy()
    delta[0, 0] = F(0.001)
    solve(state, zero, zero, inv, inv_i, 1, 0.5, 0.5, delta, zero)
    assert state["broken"] and np.all(state["impulse"][:, 1] < 0)
    assert all(v.dtype == F for v in state.values() if isinstance(v, np.ndarray))
    report = {
        "passed": True,
        "scope": "FP32 CPU translation of friction subset; physical-world angular equivalent, not instruction-identical GPU. Normal pass supplied as accumulated load. No live Colibri claim.",
        "source_p8": float(state["p8"]),
        "checks": [
            "two-anchor selection",
            "normal-only correlation",
            "broken-rebuild",
            "patch-total load split",
            "latest-header overwrite",
            "accumulated-motion target",
            "momentum/work and independent-anchor defect",
        ],
        "cases": records,
    }
    Path("/tmp/colibri_physx_patch_reference_checks.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
