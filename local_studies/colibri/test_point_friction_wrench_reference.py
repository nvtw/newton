"""Independent analytic tests for pointwise friction wrench preservation."""

import argparse
import json
import unittest
from pathlib import Path

import numpy as np

from local_studies.colibri.point_friction_wrench_reference import (
    PointFriction,
    distribute_stick,
    planar_matrix,
    stop_planar_patch,
)

REPORT = {}


def cloud(points, capacities):
    """Construct a planar fixed-normal-load reference with unit friction."""
    p = np.asarray(points, dtype=float)
    return PointFriction(p, np.tile([0.0, 0.0, 1.0], (len(p), 1)), np.asarray(capacities, dtype=float), np.ones(len(p)))


class TestFrictionWrench(unittest.TestCase):
    """Protect full pointwise wrench capacity and safe fast-stick acceptance."""

    def test_unequal_pressure_false_yaw(self):
        """Reject pooled yaw torque that the original loaded points cannot carry."""
        points = [[-0.1, 0, 0], [0.1, 0, 0]]
        full = cloud(points, [0.9, 0.1])
        pooled = cloud(points, [0.5, 0.5])
        forces = np.array([[0.0, -0.5, 0.0], [0.0, 0.5, 0.0]])
        wrench = pooled.wrench(forces)
        direction = np.array([0.0, 0.1, 0.0, 0.0, 0.0, 1.0])
        bound, _, _ = full.support(direction)
        self.assertTrue(pooled.contains_forces(forces))
        self.assertGreater(direction @ wrench, bound + 1e-12)
        self.assertAlmostEqual(bound, 0.02)
        self.assertAlmostEqual(wrench[5], 0.1)
        REPORT["unequal_pressure"] = {
            "original_pure_yaw_max": 0.02,
            "pooled_pure_yaw_max": 0.1,
            "separating_support": bound,
            "pooled_wrench": wrench.tolist(),
        }

    def test_force_torque_coupling(self):
        """Preserve the torque accompanying a saturated translation impulse."""
        full = cloud([[-0.1, 0, 0], [0.1, 0, 0]], [0.9, 0.1])
        value, forces, wrench = full.support([0, 1, 0, 0, 0, 0])
        self.assertAlmostEqual(value, 1)
        self.assertAlmostEqual(wrench[5], -0.08)
        pooled_target = np.array([0.0, 1.0, 0.0, 0.0, 0.0, 0.0])
        certificate = np.array([0.0, 0.1, 0.0, 0.0, 0.0, 1.0])
        support, _, _ = full.support(certificate)
        self.assertGreater(certificate @ pooled_target, support + 1e-12)
        self.assertTrue(full.contains_forces(forces))

    def test_center_pressure_yaw(self):
        """Retain centrally concentrated pressure when estimating torsional friction."""
        full = cloud([[-0.1, 0, 0], [0, 0, 0], [0.1, 0, 0]], [0.1, 0.8, 0.1])
        pooled = cloud([[-0.1, 0, 0], [0.1, 0, 0]], [0.5, 0.5])
        h, _, _ = full.support([0, 0, 0, 0, 0, 1])
        compressed, _, _ = pooled.support([0, 0, 0, 0, 0, 1])
        self.assertAlmostEqual(h, 0.02)
        self.assertAlmostEqual(compressed, 0.1)
        REPORT["central_pressure_yaw"] = {"original": h, "pooled": compressed}

    def test_square_combined_wrenches(self):
        """Detect both excess and lost combined wrench capacity on a uniform square."""
        a = 0.1
        full = cloud([[-a, -a, 0], [-a, a, 0], [a, -a, 0], [a, a, 0]], [0.5] * 4)
        pooled = cloud([[-a, -a, 0], [a, a, 0]], [1, 1])
        cases = []
        for linear, sign in (([a, -a, 0], -1), ([a, a, 0], 1)):
            direction = np.r_[linear, [0, 0, 1]]
            actual, _, _ = full.support(direction)
            compressed, _, _ = pooled.support(direction)
            self.assertGreater(sign * (compressed - actual), 1e-3)
            cases.append({"direction": direction.tolist(), "original_support": actual, "pooled_support": compressed})
        for direction in ([1, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 1]):
            self.assertAlmostEqual(full.support(direction)[0], pooled.support(direction)[0])
        REPORT["uniform_square"] = cases

    def test_material_and_pressure_pairing(self):
        """Prevent material averaging from changing loaded friction capacity."""
        points = np.array([[-0.1, 0, 0], [0.1, 0, 0]])
        full = PointFriction(points, np.tile([0.0, 0.0, 1.0], (2, 1)), np.array([0.9, 0.1]), np.array([0.0, 1.0]))
        bound, _, _ = full.support([0, 1, 0, 0, 0, 0])
        naive_mean = float(np.mean(full.coefficients) * sum(full.normal_impulses))
        self.assertAlmostEqual(bound, 0.1)
        self.assertAlmostEqual(naive_mean, 0.5)
        REPORT["mixed_materials"] = {"original_force_capacity": bound, "naive_averaged_capacity": naive_mean}

    def test_weighted_minimum_norm_false_rejection(self):
        """Treat a failed fast proposal as unknown despite an explicit feasible witness."""
        points = np.array([[-1.0, 0, 0], [0, 0, 0], [1, 0, 0]])
        c = np.array([0.1, 1.0, 0.1])
        target = np.array([0.0, 0.8, 0.15])
        proposed, report = distribute_stick(points, c, target)
        witness = np.array([[0.0, -0.05], [0.0, 0.75], [0.0, 0.1]])
        np.testing.assert_allclose(planar_matrix(points) @ witness.ravel(), target, atol=1e-15, rtol=0)
        self.assertTrue(np.all(np.linalg.norm(witness, axis=1) <= c + 1e-15))
        self.assertFalse(report["accepted"])
        self.assertGreater(report["max_original_cone_excess"], 0.04)
        REPORT["safe_false_rejection"] = {**report, "proposal": proposed.tolist(), "feasible_witness": witness.tolist()}

    def test_fast_stick_certified(self):
        """Accept a physical stopping proposal only after every original cone passes."""
        points = np.array([[-0.1, 0, 0], [0.1, 0, 0], [0, 0.1, 0]])
        c = np.array([1.0, 1.0, 1.0])
        h = np.diag([0.5, 0.5, 2.0])
        velocity = np.array([0.01, 0.02, 0.005])
        force, report = stop_planar_patch(points, c, h, velocity)
        self.assertTrue(report["accepted"])
        self.assertLess(report["physical_friction_work"], 0)
        np.testing.assert_allclose(
            planar_matrix(points) @ force.ravel(), report["requested_wrench"], rtol=0, atol=1e-12
        )
        zero, failed = distribute_stick(points, np.zeros(3), [0, 0, 0.1])
        self.assertIsNone(zero)
        self.assertFalse(failed["accepted"])
        REPORT["accepted_stick"] = report

    def test_external_force_and_torque_balance(self):
        """Balance nonzero external wrench with original point cones and work."""
        points = np.array([[-0.1, -0.1, 0], [-0.1, 0.1, 0], [0.1, -0.1, 0], [0.1, 0.1, 0]])
        normal = np.full(4, 0.1)
        capacity = 0.5 * normal
        mobility = np.diag([0.5, 0.5, 2.0])
        external = np.array([0.03, 0.02, 0.001])
        velocity = mobility @ external
        force, report = stop_planar_patch(points, capacity, mobility, velocity)
        self.assertTrue(report["accepted"])
        friction = planar_matrix(points) @ force.ravel()
        np.testing.assert_allclose(friction + external, 0, atol=1e-14)
        np.testing.assert_allclose(velocity + mobility @ friction, 0, atol=1e-14)
        self.assertTrue(np.all(np.linalg.norm(force, axis=1) <= capacity))
        self.assertAlmostEqual(float(normal.sum()) - 0.4, 0)
        external_work = 0.5 * float(external @ mobility @ external)
        friction_work = float(friction @ velocity + 0.5 * friction @ mobility @ friction)
        self.assertLess(friction_work, 0)
        self.assertAlmostEqual(external_work + friction_work, 0, places=16)
        REPORT["external_balance"] = {
            "external_wrench": external.tolist(),
            "friction_wrench": friction.tolist(),
            "normal_support": float(normal.sum()),
            "external_work": external_work,
            "friction_work": friction_work,
            "net_work": external_work + friction_work,
        }

    def test_shared_point_momentum_and_work(self):
        """Conserve paired angular impulse and verify dissipative physical work."""
        patch = cloud([[-0.1, -0.1, 0], [-0.1, 0.1, 0], [0.1, -0.1, 0], [0.1, 0.1, 0]], [1e-4] * 4)
        positions = np.array([[-0.3, 0.2, 0.1], [0.4, -0.2, 0.3]])
        masses = np.array([2.0, 3.0])
        inertias = np.array([np.diag([0.2, 0.3, 0.4]), np.diag([0.5, 0.6, 0.7])])
        before = np.array([[[0.01, -0.02, 0.03], [0.1, 0.02, -0.03]], [[-0.02, 0.04, 0.01], [-0.03, 0.04, 0.05]]])
        relative = np.r_[
            before[1, 0] - np.cross(before[1, 1], positions[1]) - before[0, 0] + np.cross(before[0, 1], positions[0]),
            before[1, 1] - before[0, 1],
        ]
        _, forces, _ = patch.support(-relative)
        impulse = np.zeros((2, 2, 3))
        for point, force in zip(patch.points, forces, strict=True):
            impulse[0, 0] -= force
            impulse[1, 0] += force
            impulse[0, 1] -= np.cross(point - positions[0], force)
            impulse[1, 1] += np.cross(point - positions[1], force)
        after = before.copy()
        for body in range(2):
            after[body, 0] += impulse[body, 0] / masses[body]
            after[body, 1] += np.linalg.solve(inertias[body], impulse[body, 1])
        dp = impulse[:, 0].sum(axis=0)
        dl = (impulse[:, 1] + np.cross(positions, impulse[:, 0])).sum(axis=0)
        work = float(np.sum(impulse * (before + after) * 0.5))
        energy = 0.0
        for body in range(2):
            energy += 0.5 * masses[body] * (after[body, 0] @ after[body, 0] - before[body, 0] @ before[body, 0])
            energy += 0.5 * (
                after[body, 1] @ inertias[body] @ after[body, 1] - before[body, 1] @ inertias[body] @ before[body, 1]
            )
        np.testing.assert_allclose(dp, 0, atol=1e-18)
        np.testing.assert_allclose(dl, 0, atol=1e-18)
        self.assertAlmostEqual(work, energy, places=16)
        self.assertLess(work, 0)
        independent_defect = np.cross([0, 0, 0.001], [1, 0, 0])
        np.testing.assert_array_equal(independent_defect, [0, 0.001, 0])
        REPORT["physical_ledger"] = {
            "linear_error": dp.tolist(),
            "angular_error": dl.tolist(),
            "work": work,
            "delta_KE": energy,
            "independent_endpoints_angular_defect": independent_defect.tolist(),
        }


def main():
    """Run all reference cases and optionally expose blind compression as failing."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--assume-pooled-exact", action="store_true")
    args = parser.parse_args()
    result = unittest.TextTestRunner(verbosity=2).run(
        unittest.defaultTestLoader.loadTestsFromTestCase(TestFrictionWrench)
    )
    assert result.testsRun == 9 and result.wasSuccessful()
    Path("/tmp/point_friction_wrench_reference.json").write_text(json.dumps(REPORT, indent=2))
    if args.assume_pooled_exact:
        r = REPORT["unequal_pressure"]
        assert r["pooled_pure_yaw_max"] <= r["original_pure_yaw_max"], (
            "Pooled anchors exceed true physical yaw capacity by5x"
        )


if __name__ == "__main__":
    main()
