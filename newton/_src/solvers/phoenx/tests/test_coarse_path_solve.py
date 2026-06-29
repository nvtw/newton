# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""CUDA graph tests for the experimental PhoenX coarse path correction."""

import unittest

import numpy as np
import warp as wp

from newton._src.solvers.phoenx.articulations.device import ArticulationDeviceSystem
from newton._src.solvers.phoenx.articulations.symbolic import compute_block_sparse_symbolic
from newton._src.solvers.phoenx.benchmarks.experimental.coarse_path_solve import CoarsePathSolver
from newton._src.solvers.phoenx.constraints.constraint_joint import JOINT_MODE_REVOLUTE
from newton._src.solvers.phoenx.tests.test_articulation_dvi import _make_adbs_world


class TestPhoenXCoarsePathSolve(unittest.TestCase):
    def test_operator_and_momentum_under_cuda_graph(self):
        device = wp.get_preferred_device()
        if not device.is_cuda:
            self.skipTest("coarse path tests require CUDA graph capture")

        path_count = 2
        blocks_per_path = 8
        joint_count = path_count * blocks_per_path
        body1 = np.concatenate([np.arange(1, 9, dtype=np.int32), np.arange(10, 18, dtype=np.int32)])
        body2 = body1 + 1
        positions = np.zeros((19, 3), dtype=np.float32)
        positions[1:10, 0] = np.arange(9)
        positions[10:19, 0] = np.arange(9)
        positions[10:19, 1] = 2.0
        world = _make_adbs_world(
            device,
            body1,
            body2,
            np.full(joint_count, int(JOINT_MODE_REVOLUTE), dtype=np.int32),
            positions_np=positions,
            world_kwargs={"cache_articulation_topology": True, "gravity": (0.0, 0.0, 0.0)},
        )

        rng = np.random.default_rng(17)
        velocity = np.zeros_like(positions)
        angular_velocity = np.zeros_like(positions)
        velocity[1:] = rng.normal(size=(18, 3))
        angular_velocity[1:] = rng.normal(size=(18, 3))
        world.bodies.velocity.assign(velocity)
        world.bodies.angular_velocity.assign(angular_velocity)

        topology = world.articulation_topology
        self.assertIsNotNone(topology)
        symbolic = compute_block_sparse_symbolic(
            topology.active_body1,
            topology.active_body2,
            topology.active_row_counts,
            use_meca=False,
        )
        system = ArticulationDeviceSystem.from_topology(topology, device, symbolic)
        coarse = CoarsePathSolver(
            joint_count,
            rows=5,
            color_sweeps=16,
            device=device,
            path_count=path_count,
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
        np.testing.assert_allclose(linear_after, linear_before, rtol=2.0e-6, atol=2.0e-6)
        np.testing.assert_allclose(angular_after, angular_before, rtol=2.0e-6, atol=2.0e-6)
        self.assertTrue(np.isfinite(velocity_after).all())
        self.assertTrue(np.isfinite(angular_velocity_after).all())

        rows = 5
        fine_matrix = np.zeros((joint_count * rows, joint_count * rows), dtype=np.float64)
        fine_diag = system.block_diag.numpy()[:joint_count, :rows, :rows]
        fine_off = system.block_off.numpy()[: joint_count - 1, :rows, :rows]
        for block in range(joint_count):
            row = slice(block * rows, (block + 1) * rows)
            fine_matrix[row, row] = fine_diag[block]
        for path in range(path_count):
            for local in range(blocks_per_path - 1):
                block = path * blocks_per_path + local
                edge = path * (blocks_per_path - 1) + local
                row = slice((block + 1) * rows, (block + 2) * rows)
                col = slice(block * rows, (block + 1) * rows)
                fine_matrix[row, col] = fine_off[edge]
                fine_matrix[col, row] = fine_off[edge].T

        coarse_blocks_per_path = (blocks_per_path + 2) // 2
        coarse_blocks = path_count * coarse_blocks_per_path
        prolongation = np.zeros((joint_count * rows, coarse_blocks * rows))
        coarse_nodes = list(range(0, blocks_per_path, 2))
        if coarse_nodes[-1] != blocks_per_path - 1:
            coarse_nodes.append(blocks_per_path - 1)
        for path in range(path_count):
            for local_fine in range(blocks_per_path):
                upper = int(np.searchsorted(coarse_nodes, local_fine, side="left"))
                if coarse_nodes[upper] == local_fine:
                    weights = ((upper, 1.0),)
                else:
                    lower = upper - 1
                    weight = (local_fine - coarse_nodes[lower]) / (coarse_nodes[upper] - coarse_nodes[lower])
                    weights = ((lower, 1.0 - weight), (upper, weight))
                fine = path * blocks_per_path + local_fine
                for row in range(rows):
                    for local_coarse, weight in weights:
                        block = path * coarse_blocks_per_path + local_coarse
                        prolongation[fine * rows + row, block * rows + row] = weight
        expected = prolongation.T @ fine_matrix @ prolongation

        actual = np.zeros_like(expected)
        coarse_diag = coarse.diag.numpy()[:coarse_blocks, :rows, :rows]
        coarse_off = coarse.off.numpy()[: coarse_blocks - 1, :rows, :rows]
        for block in range(coarse_blocks):
            row = slice(block * rows, (block + 1) * rows)
            actual[row, row] = coarse_diag[block]
        for path in range(path_count):
            for local in range(coarse_blocks_per_path - 1):
                block = path * coarse_blocks_per_path + local
                row = slice((block + 1) * rows, (block + 2) * rows)
                col = slice(block * rows, (block + 1) * rows)
                actual[row, col] = coarse_off[block]
                actual[col, row] = coarse_off[block].T
        np.testing.assert_allclose(actual, expected, rtol=3.0e-6, atol=3.0e-6)


if __name__ == "__main__":
    unittest.main()
