# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""CPU schedule and physical copy-algebra invariants before CUDA integration."""

import unittest

import numpy as np

from local_studies.colibri.slab_schedule import build_schedule, project_scalar_rows, validate_schedule


class TestSlabSchedule(unittest.TestCase):
    def test_repeated_edges_and_recycling(self):
        """Retain repeated physical rows and rebuild copies when topology shrinks."""
        endpoints = np.array([[0, 1]] * 17 + [[1, 2], [2, 3], [-1, 3]])
        for width in (1, 2, 8):
            schedule = build_schedule(endpoints, 5, width)
            validate_schedule(schedule, endpoints)
            self.assertEqual(len(schedule.body_slabs[4]), 0)
            self.assertGreaterEqual(len(schedule.colors), 18)
            short = build_schedule(endpoints[-2:], 5, width)
            self.assertEqual(short.body_slabs[0], [])
            self.assertEqual(short.body_slabs[1], [])

    def test_unequal_copy_counts_momentum_and_energy(self):
        """Average unequal slab counts without changing physical momentum."""
        rng = np.random.default_rng(7)
        edges = np.array([[0, i] for i in range(1, 10)] + [[1, 2], [3, 4], [2, 8]])
        mass = np.exp(rng.uniform(-8, 8, 10))
        velocity = rng.normal(size=10)
        momentum = mass @ velocity
        energy = 0.5 * mass @ velocity**2
        for width in (1, 2, 8):
            for order in (None, list(reversed(range(len(edges))))):
                schedule = build_schedule(edges, 10, width, order)
                result = project_scalar_rows(schedule, edges, 1 / mass, velocity, sweeps=8)
                self.assertAlmostEqual(mass @ result, momentum, delta=1e-11 * max(1, abs(momentum)))
                self.assertLessEqual(0.5 * mass @ result**2, energy + 1e-11)

    def test_one_slab_matches_sequential_projection(self):
        """A single shared copy reproduces ordinary sequential row projection."""
        edges = np.array([[0, 1], [1, 2], [0, 2]])
        inverse_mass = np.array([0.001, 1.0, 0.001])
        initial = np.array([1.0, 0.0, 0.0])
        schedule = build_schedule(edges, 3, colors_per_slab=8)
        expected = initial.copy()
        for a, b in edges:
            impulse = (expected[b] - expected[a]) / (inverse_mass[a] + inverse_mass[b])
            expected[a] += inverse_mass[a] * impulse
            expected[b] -= inverse_mass[b] * impulse
        np.testing.assert_array_equal(project_scalar_rows(schedule, edges, inverse_mass, initial), expected)


if __name__ == "__main__":
    unittest.main()
