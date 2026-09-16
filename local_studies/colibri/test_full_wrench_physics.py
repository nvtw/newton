"""Independent full six-coordinate contact proposal physics fixtures.

Offline CPU tests invoke the actual FP32 proposal. The NumPy float64 ledger is
an independent oracle, not a production precision requirement. All corrections
are unbiased; no claim about dissipation of positional recovery is made.
"""

import unittest

import numpy as np

from local_studies.colibri.check_full_wrench_fp32 import proposal


def point_map(points, normals):
    """Map every original normal/tangent component to a common-origin wrench."""
    matrices = []
    bases = []
    for point, original_normal in zip(points, normals, strict=True):
        normal = np.array(original_normal, dtype=float, copy=True)
        normal /= np.linalg.norm(normal)
        axis = np.eye(3)[np.argmin(abs(normal))]
        tangent = np.cross(normal, axis)
        tangent /= np.linalg.norm(tangent)
        basis = np.column_stack((normal, tangent, np.cross(normal, tangent)))
        matrices.append(np.vstack((basis, np.cross(np.tile(point, (3, 1)), basis.T).T)))
        bases.append(basis)
    return np.concatenate(matrices, axis=1), np.asarray(bases)


def physical_response(masses, inertia, positions):
    """Build a physical relative spatial mobility for two moving bodies."""
    maps = np.zeros((2, 6, 6))
    inverse_mass = np.zeros((2, 6))
    for body in range(2):
        maps[body] = np.eye(6)
        maps[body, 3:, :3] = np.cross(np.tile(-positions[body], (3, 1)), np.eye(3)).T
        inverse_mass[body] = np.r_[np.full(3, 1 / masses[body]), 1 / inertia[body]]
    h = sum(b.T @ (w[:, None] * b) for b, w in zip(maps, inverse_mass, strict=True))
    return h, maps, inverse_mass


def propose(points, normals, weights, requested, h):
    """Call the FP32 implementation with a physically constructed current state."""
    a, bases = point_map(points, normals)
    old = np.zeros((len(points), 3))
    old[:, 0] = weights
    current_velocity = h @ (a @ old.ravel() - requested)
    fixture = {
        "eligible": np.arange(len(points)),
        "point_map": a,
        "impulses": old,
        "mobility": h,
        "velocity": current_velocity,
    }
    proposed, report = proposal(fixture)
    return proposed, a, bases, report


class TestFullWrenchPhysics(unittest.TestCase):
    """Full point cones, physical response and paired momentum/work."""

    def certify(self, points, normals, weights, coefficients, requested, masses=(1.0, 2.0)):
        """Check all original rows using an independent common-point body ledger."""
        points = np.asarray(points, float)
        positions = np.array([[-0.3, 0.2, 0.4], [0.4, -0.2, -0.3]])
        masses = np.asarray(masses, float)
        inertia = masses[:, None] * np.array([[0.2, 0.3, 0.4], [0.5, 0.6, 0.7]])
        h, maps, inverse_mass = physical_response(masses, inertia, positions)
        impulse, a, bases, _ = propose(points, normals, weights, requested, h)
        self.assertEqual(impulse.dtype, np.float32)
        self.assertTrue(np.all(np.isfinite(impulse)))
        self.assertTrue(np.all(impulse[:, 0] >= 0))
        excess = np.linalg.norm(impulse[:, 1:].astype(float), axis=1) - coefficients * impulse[:, 0]
        self.assertLessEqual(float(excess.max()), 1e-7)
        actual = a @ impulse.astype(float).ravel()
        np.testing.assert_allclose(actual, requested, rtol=0, atol=2e-6)
        # The known free state cancels under the requested common-origin wrench.
        before = np.array(
            [
                inverse_mass[0] * (maps[0] @ requested),
                -inverse_mass[1] * (maps[1] @ requested),
            ]
        )
        forces = np.einsum("nij,nj->ni", bases, impulse.astype(float))
        applied = np.array(
            [
                np.r_[-forces.sum(axis=0), np.cross(points - positions[0], -forces).sum(axis=0)],
                np.r_[forces.sum(axis=0), np.cross(points - positions[1], forces).sum(axis=0)],
            ]
        )
        after = before + inverse_mass * applied
        relative = maps[1].T @ after[1] - maps[0].T @ after[0]
        np.testing.assert_allclose(relative, 0, atol=4e-6)
        # Original contact row velocities, including arbitrary normal directions.
        np.testing.assert_allclose(a.T @ relative, 0, atol=5e-6)
        dp = applied[:, :3].sum(axis=0)
        dl = (applied[:, 3:] + np.cross(positions, applied[:, :3])).sum(axis=0)
        np.testing.assert_allclose(dp, 0, atol=2e-14)
        np.testing.assert_allclose(dl, 0, atol=2e-14)
        work = float(np.sum(0.5 * applied * (before + after)))
        energy = float(np.sum(0.5 * (after**2 - before**2) / inverse_mass))
        self.assertAlmostEqual(work, energy, places=12)
        self.assertLess(work, 0)
        return impulse, a

    def test_raised_horizontal_load_requires_normal_transfer(self):
        """A raised force requires support redistribution as well as friction."""
        points = np.array([[-0.1, -0.1, 0], [-0.1, 0.1, 0], [0.1, -0.1, 0], [0.1, 0.1, 0]])
        normals = np.tile([0.0, 0.0, 1.0], (4, 1))
        external = np.array([0.2, 0, -1, 0, 0.04, 0])
        impulse, a = self.certify(points, normals, np.full(4, 0.25), np.full(4, 0.6), -external)
        np.testing.assert_allclose(impulse[:, 0], [0.15, 0.15, 0.35, 0.35], atol=2e-6)
        frozen = impulse.copy()
        frozen[:, 0] = 0.25
        self.assertAlmostEqual(float((a @ frozen.ravel() + external)[4]), 0.04, places=6)

    def test_high_mass_ratio_pair(self):
        """Retain both dynamic endpoints at 400:1 mass and inertia ratio."""
        points = np.array([[-0.2, -0.1, 0], [-0.2, 0.1, 0], [0.2, -0.1, 0], [0.2, 0.1, 0]])
        normals = np.tile([0.0, 0.0, 1.0], (4, 1))
        self.certify(
            points,
            normals,
            np.full(4, 0.25),
            np.full(4, 0.8),
            np.array([-0.03, 0.02, 1, 0.01, -0.02, 0.005]),
            masses=(1.0, 400.0),
        )

    def test_general_normals_and_materials(self):
        """Use the original normal and circular cone for every nonplanar point."""
        points = np.array([[-0.2, -0.1, 0.01], [-0.2, 0.1, -0.01], [0.2, -0.1, 0.02], [0.2, 0.1, 0]])
        normals = np.array([[0.1, 0, 1], [0, -0.1, 1], [-0.1, 0.1, 1], [0.05, 0.05, 1.0]])
        a, _ = point_map(points, normals)
        weights = np.array([0.2, 0.3, 0.25, 0.25])
        known = np.zeros((4, 3))
        known[:, 0] = weights
        self.certify(points, normals, weights, np.array([0.7, 0.8, 0.9, 1.0]), a @ known.ravel())

    def test_decisive_last_point_beyond_128(self):
        """The final contact supplies a spatial rank missing from every prefix."""
        for count in (129, 513):
            with self.subTest(count=count):
                points = np.zeros((count, 3))
                points[: count - 1, 0] = np.linspace(-0.2, 0.2, count - 1)
                points[-1] = [0, 0.2, 0]
                normals = np.tile([0.0, 0.0, 1.0], (count, 1))
                weights = np.full(count, 1 / count)
                a, _ = point_map(points, normals)
                known = np.zeros((count, 3))
                known[:, 0] = weights
                requested = a @ known.ravel()
                impulse, _ = self.certify(points, normals, weights, np.ones(count), requested)
                self.assertGreater(float(impulse[-1, 0]), 0.5 / count)
                # The missing last point removes roll torque from the normal rows.
                self.assertEqual(float(np.max(abs(a[3, :-3:3]))), 0.0)
                self.assertGreater(float(requested[3]), 0.0)

    def test_mixed_material_cone_rejects_unphysical_distribution(self):
        """A wrench fit alone must not accept a frictionless loaded contact."""
        points = np.array([[-0.1, -0.1, 0], [-0.1, 0.1, 0], [0.1, -0.1, 0], [0.1, 0.1, 0]])
        normals = np.tile([0.0, 0.0, 1.0], (4, 1))
        impulse, a, _, _ = propose(points, normals, np.full(4, 0.25), np.array([0.1, 0, 1, 0, 0, 0]), np.eye(6))
        coefficients = np.array([0.0, 0.8, 0.8, 0.8])
        np.testing.assert_allclose(a @ impulse.ravel(), [0.1, 0, 1, 0, 0, 0], atol=2e-6)
        excess = np.linalg.norm(impulse[:, 1:], axis=1) - coefficients * impulse[:, 0]
        self.assertGreater(float(excess[0]), 0.01)


if __name__ == "__main__":
    unittest.main()
