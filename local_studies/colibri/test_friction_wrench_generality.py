"""Check friction-wrench invariants independently of contact count and frame."""

import unittest

import numpy as np

from local_studies.colibri.point_friction_wrench_reference import PointFriction


class TestFrictionWrenchGenerality(unittest.TestCase):
    """Protect the physical reference used to evaluate patch implementations."""

    def test_contact_count_and_subdivision(self):
        """Preserve wrench capacity when a point's load is subdivided in place."""
        rng = np.random.default_rng(791)
        for count in (1, 2, 4, 127, 128, 129, 513):
            with self.subTest(count=count):
                points = rng.normal(size=(count, 3)) * 0.1
                normals = rng.normal(size=(count, 3))
                normals /= np.linalg.norm(normals, axis=1)[:, None]
                loads = rng.random(count) / count
                coefficients = rng.random(count)
                coefficients[::3] = 0.0
                patch = PointFriction(points, normals, loads, coefficients)
                split = PointFriction(
                    np.repeat(points, 3, axis=0),
                    np.repeat(normals, 3, axis=0),
                    np.repeat(loads / 3, 3),
                    np.repeat(coefficients, 3),
                )
                for direction in rng.normal(size=(8, 6)):
                    bound, forces, wrench = patch.support(direction)
                    split_bound, split_forces, split_wrench = split.support(direction)
                    self.assertTrue(patch.contains_forces(forces))
                    self.assertTrue(split.contains_forces(split_forces))
                    np.testing.assert_allclose(split_bound, bound, atol=2e-15, rtol=2e-14)
                    np.testing.assert_allclose(split_wrench, wrench, atol=2e-15, rtol=2e-14)

    def test_rigid_frame_covariance(self):
        """Transform force, torque and virtual velocity without changing work."""
        rng = np.random.default_rng(871)
        rotation, _ = np.linalg.qr(rng.normal(size=(3, 3)))
        rotation[:, 0] *= np.linalg.det(rotation)
        shift = np.array([0.4, -0.2, 0.7])
        count = 257
        points = rng.normal(size=(count, 3)) * 0.1
        normals = rng.normal(size=(count, 3))
        normals /= np.linalg.norm(normals, axis=1)[:, None]
        patch = PointFriction(points, normals, rng.random(count) / count, rng.random(count))
        moved = PointFriction(
            points @ rotation.T + shift, normals @ rotation.T, patch.normal_impulses, patch.coefficients
        )
        for direction in rng.normal(size=(12, 6)):
            angular = rotation @ direction[3:]
            transformed = np.r_[rotation @ direction[:3] - np.cross(angular, shift), angular]
            bound, forces, wrench = patch.support(direction)
            moved_bound, moved_forces, moved_wrench = moved.support(transformed)
            force = rotation @ wrench[:3]
            expected = np.r_[force, rotation @ wrench[3:] + np.cross(shift, force)]
            np.testing.assert_allclose(moved_bound, bound, rtol=2e-14, atol=2e-15)
            np.testing.assert_allclose(moved_forces, forces @ rotation.T, rtol=2e-13, atol=2e-15)
            np.testing.assert_allclose(moved_wrench, expected, rtol=2e-13, atol=2e-15)

    def test_single_point_has_no_free_torsion(self):
        """Reject an invented torsional moment at a single contact point."""
        point = np.array([[0.2, -0.3, 0.1]])
        normal = np.array([[0.0, 0.0, 1.0]])
        patch = PointFriction(point, normal, np.array([2.0]), np.array([0.6]))
        angular = normal[0]
        direction = np.r_[-np.cross(angular, point[0]), angular]
        bound, forces, wrench = patch.support(direction)
        self.assertEqual(bound, 0.0)
        np.testing.assert_array_equal(forces, 0.0)
        np.testing.assert_array_equal(wrench, 0.0)
        invented_torsion = np.r_[np.zeros(3), angular * 1e-3]
        self.assertGreater(direction @ invented_torsion, bound)

    def test_horizontal_load_requires_normal_transfer(self):
        """Balance a raised horizontal force with friction and redistributed support."""
        half_width = 0.1
        height = 0.2
        weight = 1.0
        applied_force = 0.3
        coefficient = 0.6
        points = np.array([[-half_width, 0.0, 0.0], [half_width, 0.0, 0.0]])
        loads = np.array(
            [(weight - height * applied_force / half_width) / 2, (weight + height * applied_force / half_width) / 2]
        )
        patch = PointFriction(points, np.tile([0.0, 0.0, 1.0], (2, 1)), loads, np.full(2, coefficient))
        tangents = np.zeros((2, 3))
        tangents[:, 0] = -applied_force * loads / weight
        self.assertTrue(patch.contains_forces(tangents))
        contact_forces = tangents.copy()
        contact_forces[:, 2] = loads
        external_force = np.array([applied_force, 0.0, -weight])
        external_torque = np.cross([0.0, 0.0, height], external_force)
        np.testing.assert_allclose(patch.wrench(contact_forces) + np.r_[external_force, external_torque], 0, atol=1e-16)
        frozen_normal_forces = contact_forces.copy()
        frozen_normal_forces[:, 2] = weight / 2
        frozen_residual = patch.wrench(frozen_normal_forces) + np.r_[external_force, external_torque]
        self.assertAlmostEqual(frozen_residual[4], height * applied_force)
        # A force beyond the tipping threshold has no nonnegative support solution.
        tipping_force = 1.01 * half_width * weight / height
        left_load = (weight - height * tipping_force / half_width) / 2
        self.assertLess(left_load, 0)


if __name__ == "__main__":
    unittest.main()
