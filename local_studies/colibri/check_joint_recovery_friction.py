"""Analytical support test for restored joint recovery in friction's trial."""

import argparse
import json
from pathlib import Path

import numpy as np
import warp as wp

from newton._src.solvers.phoenx.constraints.contact_projection import contact_project_friction_metric


@wp.kernel(enable_backward=False)
def tangent_trial(vt: wp.float32, radius: wp.float32, result: wp.array[wp.vec2f]):
    """Invoke the actual metric projector with the constrained base mobility."""
    result[0] = contact_project_friction_metric(
        wp.float32(0.5),
        wp.float32(0.0),
        wp.float32(1.0),
        vt,
        wp.float32(0.0),
        wp.float32(0.0),
        wp.float32(0.0),
        radius,
        radius,
    )


def main():
    """Require static support while preserving the requested relative recovery."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--legacy", action="store_true")
    args = parser.parse_args()
    mass = np.eye(12)
    b = np.zeros((1, 12))
    b[0, 0] = -1.0
    b[0, 6] = 1.0
    c = np.zeros((3, 12))
    c[0, 2] = 1.0
    c[1, 0] = 1.0
    c[2, 1] = 1.0
    recovery = 1e-4
    vb = (b.T @ np.linalg.solve(b @ b.T, np.array([recovery]))).ravel()
    p = mass - b.T @ np.linalg.solve(b @ b.T, b)
    normal = 1e-3
    mu = 0.5
    physical = np.zeros(12)
    physical[2] = -normal
    # Existing normal solve balances known gravity independently of x recovery.
    physical += p @ c[0] * normal
    observed = []
    for restored in (False, True):
        rhs = float(c[1] @ physical + (c[1] @ vb if restored else 0.0))
        result = wp.zeros(1, dtype=wp.vec2f, device="cpu")
        wp.launch(tangent_trial, dim=1, inputs=[wp.float32(rhs), wp.float32(mu * normal), result], device="cpu")
        lt = float(result.numpy()[0, 0])
        delta_contact = c[1] * lt
        delta_joint = -b.T @ np.linalg.solve(b @ b.T, b @ delta_contact)
        delta = delta_contact + delta_joint
        initial = physical + vb
        final = initial + delta
        joint_error = float(abs((b @ final)[0] - recovery))
        momentum = final[:3] + final[6:9] - (initial[:3] + initial[6:9]) - delta_contact[:3]
        angular = np.cross(np.array([0.1, 0.0, 0.0]), delta[6:9])
        work_contact = float(delta_contact @ ((initial + final) * 0.5))
        work_joint = float(delta_joint @ ((initial + final) * 0.5))
        energy = float(0.5 * (final @ final - initial @ initial))
        assert joint_error < 1e-12 and np.max(abs(momentum)) < 1e-12 and np.max(abs(angular)) < 1e-12
        assert abs(energy - work_contact - work_joint) < 1e-16
        assert abs(lt) <= mu * normal and work_contact <= 1e-16
        observed.append(
            {
                "restored_in_trial": restored,
                "base_vx": float(final[0]),
                "child_vx": float(final[6]),
                "tangent_impulse": lt,
                "joint_error": joint_error,
                "contact_work": work_contact,
                "joint_recovery_work": work_joint,
                "kinetic_change": energy,
                "linear_momentum_error": float(np.max(abs(momentum))),
                "angular_momentum_error": float(np.max(abs(angular))),
            }
        )
    Path("/tmp/colibri_joint_recovery_friction_analytic.json").write_text(json.dumps(observed, indent=2))
    chosen = observed[0 if args.legacy else 1]
    print(json.dumps(observed, indent=2))
    assert abs(chosen["base_vx"]) < 1e-10, "Static friction must hold the base in this feasible recovery state"


if __name__ == "__main__":
    main()
