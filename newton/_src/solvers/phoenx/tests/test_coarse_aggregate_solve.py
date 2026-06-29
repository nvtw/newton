# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""CUDA graph tests for the PhoenX branched aggregate correction."""

import unittest

import numpy as np
import warp as wp

from newton._src.solvers.phoenx.articulations.coarse_aggregate import (
    CoarseAggregateSolver,
    parent_aggregate_mapping,
)
from newton._src.solvers.phoenx.articulations.device import ArticulationDeviceSystem
from newton._src.solvers.phoenx.articulations.symbolic import compute_block_sparse_symbolic
from newton._src.solvers.phoenx.benchmarks.experimental.analyze_pgs_branched_tree import _tree_edges
from newton._src.solvers.phoenx.constraints.constraint_joint import JOINT_MODE_BALL_SOCKET
from newton._src.solvers.phoenx.tests.test_articulation_dvi import _make_adbs_world


class TestPhoenXCoarseAggregateSolve(unittest.TestCase):
    def test_operator_and_momentum_under_cuda_graph(self):
        device = wp.get_preferred_device()
        if not device.is_cuda:
            self.skipTest("coarse aggregate tests require CUDA graph capture")

        raw_body1, raw_body2, depth = _tree_edges(8, 3, 6)
        body1 = raw_body1 + 1
        body2 = raw_body2 + 1
        body_count = int(body2.max()) + 1
        positions = np.zeros((body_count, 3), dtype=np.float32)
        positions[1:, 0] = np.arange(body_count - 1)
        world = _make_adbs_world(
            device,
            body1,
            body2,
            np.full(body1.size, int(JOINT_MODE_BALL_SOCKET), dtype=np.int32),
            positions_np=positions,
            world_kwargs={"cache_articulation_topology": True, "gravity": (0.0, 0.0, 0.0)},
        )
        rng = np.random.default_rng(29)
        velocity = np.zeros_like(positions)
        angular_velocity = np.zeros_like(positions)
        velocity[1:] = rng.normal(size=(body_count - 1, 3))
        angular_velocity[1:] = rng.normal(size=(body_count - 1, 3))
        world.bodies.velocity.assign(velocity)
        world.bodies.angular_velocity.assign(angular_velocity)

        topology = world.articulation_topology
        symbolic = compute_block_sparse_symbolic(
            topology.active_body1,
            topology.active_body2,
            topology.active_row_counts,
            use_meca=False,
        )
        system = ArticulationDeviceSystem.from_topology(topology, device, symbolic)
        mapping = parent_aggregate_mapping(raw_body1, raw_body2, depth)
        coarse = CoarseAggregateSolver(
            mapping,
            symbolic.n_off_row_idx[: symbolic.nnz_n],
            symbolic.n_off_col_idx[: symbolic.nnz_n],
            rows=3,
            color_sweeps=16,
            device=device,
        )

        linear_before = velocity[1:].sum(axis=0)
        angular_before = (angular_velocity[1:] + np.cross(positions[1:], velocity[1:])).sum(axis=0)
        with wp.ScopedCapture(device=device) as capture:
            system.populate_from_adbs_constraints(world.constraints, world.bodies, dt=1.0 / 6000.0, device=device)
            system.compute_residual(
                world.bodies,
                dt=1.0 / 6000.0,
                recovery_speed=0.0,
                device=device,
            )
            system.assemble_block_sparse_matrix(
                world.bodies.inverse_mass,
                world.bodies.inverse_inertia_world,
                diagonal_regularization=0.001,
                device=device,
            )
            coarse.solve(system, device=device)
            system.apply_solution(
                world.bodies,
                world.bodies.inverse_mass,
                world.bodies.inverse_inertia_world,
                device=device,
            )
        wp.capture_launch(capture.graph)

        velocity_after = world.bodies.velocity.numpy()
        angular_velocity_after = world.bodies.angular_velocity.numpy()
        linear_after = velocity_after[1:].sum(axis=0)
        angular_after = (angular_velocity_after[1:] + np.cross(positions[1:], velocity_after[1:])).sum(axis=0)
        np.testing.assert_allclose(linear_after, linear_before, rtol=3.0e-6, atol=3.0e-6)
        np.testing.assert_allclose(angular_after, angular_before, rtol=3.0e-6, atol=3.0e-6)

        rows = 3
        fine_blocks = int(body1.size)
        fine_matrix = np.zeros((fine_blocks * rows, fine_blocks * rows), dtype=np.float64)
        fine_diag = system.block_diag.numpy()[:fine_blocks, :rows, :rows]
        fine_off = system.block_off.numpy()[: symbolic.nnz_n, :rows, :rows]
        for block in range(fine_blocks):
            row = slice(block * rows, (block + 1) * rows)
            fine_matrix[row, row] = fine_diag[block]
        for edge, (row_block, col_block) in enumerate(
            zip(symbolic.n_off_row_idx[: symbolic.nnz_n], symbolic.n_off_col_idx[: symbolic.nnz_n], strict=True)
        ):
            row = slice(row_block * rows, (row_block + 1) * rows)
            col = slice(col_block * rows, (col_block + 1) * rows)
            fine_matrix[row, col] = fine_off[edge]
            fine_matrix[col, row] = fine_off[edge].T

        prolongation = np.zeros((fine_blocks * rows, coarse.coarse_blocks * rows))
        for fine, coarse_block in enumerate(mapping):
            for row in range(rows):
                prolongation[fine * rows + row, coarse_block * rows + row] = 1.0
        expected = prolongation.T @ fine_matrix @ prolongation
        actual = np.zeros_like(expected)
        coarse_diag = coarse.diag.numpy()[: coarse.coarse_blocks, :rows, :rows]
        coarse_off = coarse.off.numpy()[: len(coarse.coarse_pairs), :rows, :rows]
        for block in range(coarse.coarse_blocks):
            row = slice(block * rows, (block + 1) * rows)
            actual[row, row] = coarse_diag[block]
        for edge, (row_block, col_block) in enumerate(coarse.coarse_pairs):
            row = slice(row_block * rows, (row_block + 1) * rows)
            col = slice(col_block * rows, (col_block + 1) * rows)
            actual[row, col] = coarse_off[edge]
            actual[col, row] = coarse_off[edge].T
        np.testing.assert_allclose(actual, expected, rtol=3.0e-6, atol=3.0e-6)


if __name__ == "__main__":
    unittest.main()
