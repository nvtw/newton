"""Independent scalar/vector Coulomb reference checks."""

import unittest

import numpy as np

from local_studies.colibri.coulomb_semismooth import solve_coulomb


class TestMixedFriction(unittest.TestCase):
    def test_scalar_vector_identity(self):
        """Broadcasting one coefficient preserves the complete numerical solve."""
        A = np.diag([2.0, 3.0, 4.0, 1.0, 2.0, 3.0])
        rhs = np.array([-1.0, 2.0, -0.5, -2.0, 1.0, 0.2])
        old = np.zeros(6)
        scalar, report = solve_coulomb(A, rhs, old, np.zeros(2), 0.5)
        vector, other = solve_coulomb(A, rhs, old, np.zeros(2), np.full(2, 0.5))
        np.testing.assert_array_equal(scalar, vector)
        self.assertEqual(report, other)

    def test_mixed_independent_contacts(self):
        """Frictionless/sliding/separating contacts obey normal law and dissipate."""
        u = np.array([-1.0, 2.0, 0.0, -1.0, 2.0, 0.0, 1.0, 2.0, 0.0])
        impulse, _ = solve_coulomb(np.eye(9), u, np.zeros(9), np.zeros(3), np.array([0.0, 0.5, 0.8]))
        np.testing.assert_allclose(impulse, [1.0, 0.0, 0.0, 1.0, -0.5, 0.0, 0.0, 0.0, 0.0], atol=1e-8)
        final = u + impulse
        self.assertLessEqual(float(final @ final - u @ u), 0.0)
        np.testing.assert_allclose(impulse[::3] * final[::3], 0.0, atol=1e-8)


if __name__ == "__main__":
    unittest.main()
