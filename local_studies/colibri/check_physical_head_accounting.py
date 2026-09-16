"""CPU algebraic control for physical head followed by split overflow.

This is a proposed schedule's accounting proof, not a production regression.
All poses stay fixed; paired impulses act at shared off-center world pivots.
"""

import json
from pathlib import Path

import numpy as np


def main():
    """Check copy-count-independent head response and complete impulse budgets."""
    mass = np.array([2.0, 3.0, 5.0])
    inertia = np.array([[0.7, 0.9, 1.1], [1.2, 0.8, 1.4], [1.8, 1.6, 1.3]])
    positions = np.array([[-0.4, 0.1, 0.3], [0.2, -0.2, 0.1], [0.6, 0.3, -0.2]])
    metric = np.concatenate((np.repeat(mass[:, None], 3, axis=1), inertia), axis=1)
    mobility = 1.0 / metric
    initial = np.array(
        [[0.4, -0.2, 0.1, 0.3, -0.1, 0.2], [-0.3, 0.1, -0.2, -0.2, 0.4, 0.1], [0.1, -0.3, 0.2, 0.1, 0.2, -0.4]]
    )

    def row(a, b, pivot, normal):
        normal = np.asarray(normal, dtype=float)
        normal /= np.linalg.norm(normal)
        result = np.zeros((3, 6))
        for body, sign in ((a, -1), (b, 1)):
            result[body] = sign * np.r_[normal, np.cross(np.asarray(pivot) - positions[body], normal)]
        return result

    head = row(0, 1, [0.1, 0.05, 0.4], [1, 0.2, -0.1])
    tail = [row(1, 2, [0.4, 0.1, 0.2], [0.2, 1, 0.3]), row(0, 2, [0.2, 0.2, 0.1], [-0.3, 0.1, 1])]
    # Closing unilateral examples, chosen by row orientation, not a static threshold.
    for i, contact in enumerate(tail):
        if np.sum(contact * initial) > 0:
            tail[i] = -contact

    def momentum(velocity):
        p = metric * velocity
        return np.r_[p[:, :3].sum(axis=0), (p[:, 3:] + np.cross(positions, p[:, :3])).sum(axis=0)]

    def physical_projection(velocity, jacobian, unilateral=False):
        impulse = -np.sum(jacobian * velocity) / np.sum(jacobian * mobility * jacobian)
        if unilateral:
            impulse = max(0.0, impulse)
        return velocity + mobility * jacobian * impulse

    expected_head = physical_projection(initial, head)
    reports = []
    for count_values in ([1, 1, 1], [2, 2, 2], [98, 98, 98], [2, 3, 5]):
        counts = np.array(count_values)
        for physical_head in (False, True):
            copies = [np.repeat(initial[b : b + 1], counts[b], axis=0) for b in range(3)]
            impulses = np.zeros_like(initial)

            def copied_projection(jacobian, batch, unilateral, copies=copies, counts=counts, impulses=impulses):
                velocity = np.array([copies[b][batch % counts[b]] for b in range(3)])
                inverse = counts[:, None] * mobility
                lam = -np.sum(jacobian * velocity) / np.sum(jacobian * inverse * jacobian)
                if unilateral:
                    lam = max(0.0, lam)
                for b in range(3):
                    copies[b][batch % counts[b]] += inverse[b] * jacobian[b] * lam
                impulses[:] += jacobian * lam

            if physical_head:
                lam = -np.sum(head * initial) / np.sum(head * mobility * head)
                impulses += head * lam
                for b in range(3):
                    copies[b][:] = expected_head[b]
            else:
                copied_projection(head, 0, False)
            head_velocity = np.array([v.mean(axis=0) for v in copies])
            if physical_head:
                np.testing.assert_allclose(head_velocity, expected_head, atol=2e-15, rtol=0)
            for batch, contact in enumerate(tail):
                copied_projection(contact, batch, True)
            final = np.array([v.mean(axis=0) for v in copies])
            reconstruction = initial + mobility * impulses
            np.testing.assert_allclose(final, reconstruction, atol=3e-15, rtol=0)
            defect = momentum(final) - momentum(initial)
            np.testing.assert_allclose(defect, 0, atol=5e-15, rtol=0)
            work = np.sum(impulses * (initial + final) * 0.5)
            delta_energy = 0.5 * np.sum(metric * (final * final - initial * initial))
            assert abs(work - delta_energy) < 3e-15
            assert delta_energy <= 3e-15
            if np.all(counts == 1):
                sequential = expected_head.copy()
                for contact in tail:
                    sequential = physical_projection(sequential, contact, True)
                np.testing.assert_allclose(final, sequential, atol=3e-15, rtol=0)
            reports.append(
                {
                    "counts": counts.tolist(),
                    "schedule": "physical_head" if physical_head else "global_copy",
                    "head_residual": float(np.sum(head * head_velocity)),
                    "head_difference": float(np.max(np.abs(head_velocity - expected_head))),
                    "momentum_defect": defect.tolist(),
                    "kinetic_change_J": float(delta_energy),
                    "work_balance_error_J": float(work - delta_energy),
                }
            )
    assert all(r["head_difference"] > 1e-3 for r in reports if r["schedule"] == "global_copy" and max(r["counts"]) > 1)
    output = {
        "scope": "Fixed-pose algebraic schedule proof; one bilateral row and two unilateral contacts; no production claim",
        "cases": reports,
    }
    Path("/tmp/physical_head_accounting.json").write_text(json.dumps(output, indent=2))
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
