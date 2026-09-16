"""Independent contact-friction wrench envelopes and a certified stick proposal.

Offline reference only. Normal impulses are fixed inputs, not an independent
solution of coupled normal contact. Each point retains its own Coulomb disk.
"""

from dataclasses import dataclass

import numpy as np


@dataclass
class PointFriction:
    """Physical contact application points, normals, and friction capacities."""

    points: np.ndarray
    normals: np.ndarray
    normal_impulses: np.ndarray
    coefficients: np.ndarray

    def capacities(self):
        """Return each original static Coulomb radius in N s."""
        values = np.asarray(self.normal_impulses) * np.asarray(self.coefficients)
        assert np.all(values >= 0)
        return values

    def support(self, direction):
        """Return exact support of the six-dimensional product-disk wrench set.

        direction=(linear, angular) is a virtual relative rigid velocity.
        Maximizing its work independently at every disk gives a rigorous
        separation certificate, without polygonal cone approximations.
        """
        direction = np.asarray(direction, dtype=float)
        normals = np.asarray(self.normals, dtype=float)
        np.testing.assert_allclose(np.linalg.norm(normals, axis=1), 1, rtol=0, atol=1e-14)
        velocities = direction[:3] + np.cross(direction[3:], self.points)
        tangent = velocities - normals * np.sum(normals * velocities, axis=1)[:, None]
        lengths = np.linalg.norm(tangent, axis=1)
        forces = np.divide(
            tangent * self.capacities()[:, None],
            lengths[:, None],
            out=np.zeros_like(tangent),
            where=lengths[:, None] > 0,
        )
        wrench = self.wrench(forces)
        bound = float(self.capacities() @ lengths)
        np.testing.assert_allclose(direction @ wrench, bound, rtol=2e-14, atol=1e-14)
        return bound, forces, wrench

    def wrench(self, forces):
        """Sum equal/opposite point impulses about the same world origin."""
        forces = np.asarray(forces)
        return np.r_[forces.sum(axis=0), np.cross(self.points, forces).sum(axis=0)]

    def contains_forces(self, forces, tolerance=1e-12):
        """Check original point cones and tangent planes, not only patch sum."""
        forces = np.asarray(forces)
        return bool(
            np.max(np.abs(np.sum(forces * self.normals, axis=1))) <= tolerance
            and np.all(np.linalg.norm(forces, axis=1) <= self.capacities() + tolerance)
        )


def planar_matrix(points):
    """Map all planar tangent impulses to (Fx,Fy,tau_z)."""
    points = np.asarray(points)
    assert np.max(abs(points[:, 2] - points[0, 2])) < 1e-14
    matrix = np.zeros((3, 2 * len(points)))
    matrix[0, 0::2] = 1
    matrix[1, 1::2] = 1
    matrix[2, 0::2] = -points[:, 1]
    matrix[2, 1::2] = points[:, 0]
    return matrix


def distribute_stick(points, capacities, target, tolerance=1e-10):
    """Propose weighted minimum-norm impulses; reject safely if any cone fails.

    This is a sufficient feasibility test, not a complete cone-feasibility solver.
    Positive definite 3x3 factorization failure also rejects without deleting
    physical rows or adding regularization.
    """
    a = planar_matrix(points)
    c = np.asarray(capacities, dtype=float)
    d = np.repeat(c, 2)
    gram = (a * d) @ a.T
    try:
        factor = np.linalg.cholesky(gram)
        y = np.linalg.solve(factor.T, np.linalg.solve(factor, np.asarray(target)))
    except np.linalg.LinAlgError:
        return None, {"accepted": False, "reason": "Singular weighted wrench map; use point fallback"}
    impulse = (d * (a.T @ y)).reshape(-1, 2)
    residual = float(np.max(abs(a @ impulse.ravel() - target)))
    excess = float(np.max(np.linalg.norm(impulse, axis=1) - c))
    accepted = residual <= tolerance and excess <= tolerance
    return impulse, {
        "accepted": bool(accepted),
        "wrench_residual": residual,
        "max_original_cone_excess": excess,
        "reason": "Certified original cones" if accepted else "Proposal rejected; feasibility unresolved",
    }


def stop_planar_patch(points, capacities, mobility, relative_velocity):
    """Propose a zero-bias sticking impulse using a physical 3-DOF response."""
    mobility = np.asarray(mobility, dtype=float)
    np.testing.assert_allclose(mobility, mobility.T, rtol=0, atol=1e-14)
    try:
        factor = np.linalg.cholesky(mobility)
        wrench = -np.linalg.solve(factor.T, np.linalg.solve(factor, relative_velocity))
    except np.linalg.LinAlgError:
        return None, {"accepted": False, "reason": "Unresolved physical mobility; no mode truncation"}
    impulse, report = distribute_stick(points, capacities, wrench)
    report["requested_wrench"] = wrench.tolist()
    report["stick_velocity_residual"] = float(np.max(abs(relative_velocity + mobility @ wrench)))
    report["physical_friction_work"] = float(wrench @ relative_velocity + 0.5 * wrench @ mobility @ wrench)
    if report["accepted"]:
        assert report["stick_velocity_residual"] <= 1e-10
        assert report["physical_friction_work"] <= 1e-12
    return impulse, report
