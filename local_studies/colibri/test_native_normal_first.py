# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Independent point-order controls for a rigid body on a plane."""

import ast
import unittest
from pathlib import Path

import numpy as np

from .native_normal_first import split_sweeps


def run_points(points, separated):
    """Solve unconstrained sticking tangents with all original normal capacities."""
    axes = np.array([[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    rows = [np.c_[axes, np.cross(p, axes)] for p in points]
    w = np.diag([1.0, 1.0, 1.0, 2.0, 3.0, 4.0])
    v = np.array([0.1, 0.2, -1.0, 0.2, 0.3, 0.4])
    initial = v.copy()
    impulses = np.zeros((len(points), 3))
    visits = []

    def normal(i):
        row = rows[i][0]
        old = impulses[i, 0]
        new = max(0.0, old - row @ v / (row @ w @ row))
        impulses[i, 0] = new
        v[:] += w @ row * (new - old)
        visits.append(("normal", i))

    def tangent(i):
        jac = rows[i][1:]
        delta = np.linalg.solve(jac @ w @ jac.T, -jac @ v)
        impulses[i, 1:] += delta
        v[:] += w @ jac.T @ delta
        visits.append(("tangent", i))
        assert np.linalg.norm(impulses[i, 1:]) < 100 * impulses[i, 0]

    if separated:
        for i in range(len(points)):
            normal(i)
        for i in range(len(points)):
            tangent(i)
    else:
        for i in range(len(points)):
            normal(i)
            tangent(i)
    wrench = sum((row.T @ impulse for row, impulse in zip(rows, impulses, strict=True)), np.zeros(6))
    np.testing.assert_allclose(v - initial, w @ wrench, atol=1e-14)
    return v, impulses, visits


class TestNormalFirst(unittest.TestCase):
    def test_single_point_equivalence(self):
        """Leave a single original contact unchanged when splitting its two stages."""
        a = run_points([[0.2, 0.1, 0]], False)
        b = run_points([[0.2, 0.1, 0]], True)
        np.testing.assert_array_equal(a[0], b[0])
        np.testing.assert_array_equal(a[1], b[1])

    def test_multiple_points_keep_all_capacities(self):
        """Expose order-dependent coupling without dropping any original point."""
        points = [[-0.2, 0.1, -0.3], [0.3, -0.1, -0.3]]
        a = run_points(points, False)
        b = run_points(points, True)
        self.assertEqual(b[2], [("normal", 0), ("normal", 1), ("tangent", 0), ("tangent", 1)])
        self.assertGreater(np.max(abs(a[0] - b[0])), 1e-4)
        self.assertTrue(np.all(b[1][:, 0] >= 0))

    def test_projection_source_unchanged(self):
        """Keep the exact point-friction projection and every complete point loop."""
        path = Path(__file__).resolve().parents[2] / "newton/_src/solvers/phoenx/articulations/maximal_contact_gs.py"
        text = path.read_text()
        start = text.index("def iterate_maximal_contact_runs_kernel(")
        source = text[start : text.index("\n\n@wp.kernel", start)]
        result = split_sweeps(source)
        ast.parse(result)
        marker = "                    rhs0 = tangent_velocity0"
        end = "                response.contact_active[articulation]"
        original = source[source.index(marker) : source.index(end)]
        self.assertIn(original, result)
        self.assertEqual(result.count("for scheduled in range(begin, end):"), 2)
        self.assertEqual(result.count("for offset in range(count):"), 2)
        self.assertEqual(result.count("contact_project_normal_velocity_update("), 1)


if __name__ == "__main__":
    unittest.main()
