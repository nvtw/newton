# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for PhoenX full-coordinate articulation DVI pieces."""

from __future__ import annotations

import unittest

import numpy as np
import warp as wp

from newton._src.solvers.phoenx.articulations import (
    ArticulationTopology,
    PrefactorizedArticulationSystem,
    compute_block_sparse_symbolic,
    d6_constraint_row_count,
    factorize_ldlt,
    joint_constraint_row_count,
    revolute_rows,
)
from newton._src.solvers.phoenx.body import body_container_zeros
from newton._src.solvers.phoenx.constraints.constraint_joint import (
    JOINT_MODE_BALL_SOCKET,
    JOINT_MODE_CABLE,
    JOINT_MODE_CYLINDRICAL,
    JOINT_MODE_FIXED,
    JOINT_MODE_PLANAR,
    JOINT_MODE_PRISMATIC,
    JOINT_MODE_REVOLUTE,
    JOINT_MODE_UNIVERSAL,
)
from newton._src.solvers.phoenx.solver_phoenx import PhoenXWorld


class TestPhoenXArticulationDVI(unittest.TestCase):
    def test_joint_row_counts(self):
        self.assertEqual(joint_constraint_row_count(int(JOINT_MODE_REVOLUTE)), 5)
        self.assertEqual(joint_constraint_row_count(int(JOINT_MODE_PRISMATIC)), 5)
        self.assertEqual(joint_constraint_row_count(int(JOINT_MODE_BALL_SOCKET)), 3)
        self.assertEqual(joint_constraint_row_count(int(JOINT_MODE_FIXED)), 6)
        self.assertEqual(joint_constraint_row_count(int(JOINT_MODE_CABLE)), 3)
        self.assertEqual(joint_constraint_row_count(int(JOINT_MODE_UNIVERSAL)), 4)
        self.assertEqual(joint_constraint_row_count(int(JOINT_MODE_CYLINDRICAL)), 4)
        self.assertEqual(joint_constraint_row_count(int(JOINT_MODE_PLANAR)), 3)

        self.assertEqual(d6_constraint_row_count(0, 0), 6)
        self.assertEqual(d6_constraint_row_count(1, 1), 4)
        self.assertEqual(d6_constraint_row_count(3, 3), 0)

    def test_topology_normalizes_static_bodies(self):
        topology = ArticulationTopology.from_host(
            np.array([0, 1, 2], dtype=np.int32),
            np.array([1, 2, 3], dtype=np.int32),
            np.array(
                [
                    int(JOINT_MODE_BALL_SOCKET),
                    int(JOINT_MODE_REVOLUTE),
                    int(JOINT_MODE_CABLE),
                ],
                dtype=np.int32,
            ),
            static_body_indices=np.array([0], dtype=np.int32),
        )

        np.testing.assert_array_equal(topology.body1, np.array([-1, 1, 2], dtype=np.int32))
        np.testing.assert_array_equal(topology.active_row_counts, np.array([3, 5, 3], dtype=np.int32))
        np.testing.assert_array_equal(topology.active_block_offsets, np.array([0, 3, 8, 11], dtype=np.int32))
        self.assertEqual(topology.total_rows, 11)

    def test_symbolic_two_link_chain(self):
        symbolic = compute_block_sparse_symbolic(
            np.array([-1, 1, 2], dtype=np.int32),
            np.array([1, 2, 3], dtype=np.int32),
            np.array([5, 5, 5], dtype=np.int32),
            use_meca=False,
        )

        self.assertEqual(symbolic.num_blocks, 3)
        self.assertEqual(symbolic.total_rows, 15)
        np.testing.assert_array_equal(symbolic.pivot_order, np.array([0, 1, 2], dtype=np.int32))
        self.assertEqual(symbolic.nnz_n, 2)
        self.assertEqual(symbolic.nnz_l, 2)
        self.assertEqual(symbolic.num_levels, 3)
        np.testing.assert_array_equal(symbolic.parent, np.array([1, 2, -1], dtype=np.int32))

    def test_dense_ldlt_matches_numpy_solve(self):
        rng = np.random.default_rng(7)
        a = rng.normal(size=(8, 8))
        spd = a @ a.T + np.eye(8) * 0.25
        rhs = rng.normal(size=8)

        factor = factorize_ldlt(spd)
        np.testing.assert_allclose(factor.solve(rhs), np.linalg.solve(spd, rhs), rtol=1.0e-11, atol=1.0e-11)

    def test_prefactorized_system_dense_assembly(self):
        topology = ArticulationTopology.from_host(
            np.array([0, 1], dtype=np.int32),
            np.array([1, 2], dtype=np.int32),
            np.array([int(JOINT_MODE_BALL_SOCKET), int(JOINT_MODE_BALL_SOCKET)], dtype=np.int32),
            static_body_indices=np.array([0], dtype=np.int32),
        )
        system = PrefactorizedArticulationSystem.from_topology(topology, diagonal_regularization=0.0)

        jac = np.zeros((6, 12), dtype=np.float64)
        for axis in range(3):
            jac[axis, 6 + axis] = 1.0
            jac[3 + axis, axis] = -1.0
            jac[3 + axis, 6 + axis] = 1.0
        inv_mass = np.array([0.0, 2.0, 3.0], dtype=np.float64)
        inv_inertia = np.zeros((3, 3, 3), dtype=np.float64)

        h = system.assemble_dense_matrix(jac, inv_mass, inv_inertia)
        expected = np.block(
            [
                [2.0 * np.eye(3), -2.0 * np.eye(3)],
                [-2.0 * np.eye(3), 5.0 * np.eye(3)],
            ]
        )
        np.testing.assert_allclose(h, expected)

        rhs = np.arange(1, 7, dtype=np.float64)
        system.factorize_from_jacobian(jac, inv_mass, inv_inertia)
        expected_solution = np.linalg.solve(expected, rhs)
        np.testing.assert_allclose(system.solve(rhs), expected_solution)
        np.testing.assert_allclose(system.solve_block_sparse(rhs), expected_solution)

    def test_revolute_row_builder_signs(self):
        rows = revolute_rows(
            parent_anchor_world=np.array([0.0, 0.0, 0.0]),
            child_anchor_world=np.array([0.1, 0.2, -0.3]),
            parent_com_world=np.array([0.0, -0.5, 0.0]),
            child_com_world=np.array([0.0, 0.5, 0.0]),
            parent_axis_world=np.array([0.0, 0.0, 1.0]),
            child_axis_world=np.array([0.0, 0.0, 1.0]),
        )

        self.assertEqual(rows.jacobian.shape, (5, 12))
        np.testing.assert_allclose(rows.violation[:3], np.array([0.1, 0.2, -0.3]))
        np.testing.assert_allclose(rows.jacobian[0, 0:3], np.array([-1.0, 0.0, 0.0]))
        np.testing.assert_allclose(rows.jacobian[0, 6:9], np.array([1.0, 0.0, 0.0]))
        np.testing.assert_allclose(rows.violation[3:], np.zeros(2), atol=1.0e-15)

    def test_phoenx_world_caches_prefactorized_topology(self):
        device = wp.get_preferred_device()
        bodies = body_container_zeros(3, device=device)
        bodies.inverse_mass.assign(np.array([0.0, 1.0, 1.0], dtype=np.float32))
        constraints = PhoenXWorld.make_constraint_container(num_joints=2, device=device)
        world = PhoenXWorld(
            bodies=bodies,
            constraints=constraints,
            num_joints=2,
            num_worlds=1,
            device=device,
        )

        body1 = wp.array(np.array([0, 1], dtype=np.int32), dtype=wp.int32, device=device)
        body2 = wp.array(np.array([1, 2], dtype=np.int32), dtype=wp.int32, device=device)
        mode = wp.array(
            np.array([int(JOINT_MODE_BALL_SOCKET), int(JOINT_MODE_REVOLUTE)], dtype=np.int32),
            dtype=wp.int32,
            device=device,
        )

        world._cache_prefactorized_articulation_topology(body1, body2, mode)

        self.assertIsNotNone(world.articulation_topology)
        self.assertIsNotNone(world.articulation_system)
        np.testing.assert_array_equal(world.articulation_topology.body1, np.array([-1, 1], dtype=np.int32))
        self.assertEqual(world.articulation_topology.total_rows, 8)
        self.assertEqual(world.articulation_system.symbolic.total_rows, 8)


if __name__ == "__main__":
    unittest.main()
