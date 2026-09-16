"""Analytical and impulse-ledger gates for the FP32 PhysX D6 subset."""

import copy
import json
from pathlib import Path

import numpy as np

from local_studies.colibri.physx_d6_reference import F, Row, aligned_hinge_rows, conclude, prepare, solve


def physical(v, si):
    """Convert source square-root-inertia angular coordinates to omega."""
    out = v.astype(np.float64).copy()
    for b in range(2):
        out[b, 1] = si[b].astype(float) @ v[b, 1].astype(float)
    return out


def ledger(initial, final, impulse, masses, si, positions):
    """Independently check total P/L and midpoint work using physical inertias."""
    inertia = np.array([np.linalg.inv(s.astype(float) @ s.astype(float)) for s in si])
    before, after = physical(initial, si), physical(final, si)
    change = np.zeros((2, 6))
    for b in range(2):
        change[b, :3] = masses[b] * (after[b, 0] - before[b, 0])
        change[b, 3:] = inertia[b] @ (after[b, 1] - before[b, 1])
    total = impulse.astype(float).sum(axis=0)
    momentum = change[:, :3].sum(axis=0)
    angular = sum((change[b, 3:] + np.cross(positions[b], change[b, :3]) for b in range(2)), start=np.zeros(3))
    energy = sum(
        0.5 * masses[b] * (after[b, 0] @ after[b, 0] - before[b, 0] @ before[b, 0])
        + 0.5 * (after[b, 1] @ inertia[b] @ after[b, 1] - before[b, 1] @ inertia[b] @ before[b, 1])
        for b in range(2)
    )
    work = float(np.sum(total * ((before + after) * 0.5).reshape(2, 6)))
    response = float(abs(change - total).max())
    assert response < 2e-7 and abs(momentum).max() < 2e-7 and abs(angular).max() < 2e-7
    assert abs(energy - work) < 2e-8
    return {
        "response_error": response,
        "P_error": float(abs(momentum).max()),
        "L_error": float(abs(angular).max()),
        "work_identity_error": abs(energy - work),
        "delta_KE": energy,
    }


def main():
    """Verify source force spring against its closed-form implicit step."""
    dt = F(1 / 3600)
    k, d = F(0.572957795), F(0.572957795)
    si = np.array([np.diag([2.0, 3.0, 4.0]), np.diag([5.0, 6.0, 7.0])], dtype=np.float32)
    im = np.array([0.5, 1 / 3], dtype=np.float32)
    axis = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    z = np.zeros(3, dtype=np.float32)
    row = Row(z.copy(), axis.copy(), z.copy(), axis.copy(), F(0.12), F(0), 0, True, True, k, d)
    v = np.zeros((2, 2, 3), dtype=np.float32)
    v[0, 1, 0] = F(0.07 / 2)
    v[1, 1, 0] = F(-0.02 / 5)
    block = prepare([row], im, si, dt, 1 / 120)
    after, impulse = solve(block, v, np.zeros((2, 3), dtype=np.float32))
    response = 29.0
    speed = 0.09
    exact = (
        -float(dt)
        * (float(k) * float(row.error) + (float(dt) * float(k) + float(d)) * speed)
        / (1 + float(dt) * (float(dt) * float(k) + float(d)) * response)
    )
    actual = float(block["rows"][0]["applied"])
    assert abs(actual - exact) < 2e-11
    new_speed = float(physical(after, si)[0, 1, 0] - physical(after, si)[1, 1, 0])
    constitutive = actual + float(dt) * (float(k) * (float(row.error) + float(dt) * new_speed) + float(d) * new_speed)
    assert abs(constitutive) < 2e-11
    scalar = ledger(v, after, impulse, 1 / im.astype(float), si, np.zeros((2, 3)))
    prior = after.copy()
    conclude(block)
    final, delta = solve(block, after, np.zeros((2, 3), dtype=np.float32))
    assert final.tobytes() == prior.tobytes() and not np.any(delta)
    positions = np.array([[-0.2, 0.0, 0.0], [0.3, 0.1, 0.0]], dtype=np.float32)
    pivot = np.array([0.0, 0.2, 0.05], dtype=np.float32)
    arms = pivot - positions
    rows = aligned_hinge_rows(arms, F(np.sin(-0.2 / 2)), k, d)
    block = prepare(rows, im, si, dt, 1 / 120)
    assert [entry["row"].hint for entry in block["rows"]] == [0, 1024, 1024, 2048, 2048, 2048]
    start = np.array(
        [[[0.001, -0.002, 0.003], [0.001, 0.002, -0.001]], [[-0.003, 0.002, 0.001], [-0.001, 0.001, 0.002]]],
        dtype=np.float32,
    )
    end, impulse = solve(block, start, arms)
    six = ledger(start, end, impulse, 1 / im.astype(float), si, positions.astype(float))
    # The source's finite ordered solve is not a simultaneous block solve.
    reverse = copy.deepcopy(block)
    for entry in reverse["rows"]:
        entry["applied"] = F(0)
    reverse["rows"].reverse()
    reversed_end, _ = solve(reverse, start, arms)
    order_difference = float(abs(reversed_end - end).max())
    assert order_difference > 1e-6
    report = {
        "source": "PhysX ed6e5ca2474c9c80ad4f4826591b88476779c6ef",
        "scope": "FP32 CPU prepared-row source translation; no SDK/GPU bitidentity claim",
        "analytic_spring_impulse": exact,
        "translated_impulse": actual,
        "implicit_spring_residual": constitutive,
        "scalar_ledger": scalar,
        "six_row_ledger": six,
        "reverse_order_velocity_difference": order_difference,
        "conclude_spring_velocity_bytes_unchanged": True,
    }
    Path("/tmp/physx_d6_reference_test.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
