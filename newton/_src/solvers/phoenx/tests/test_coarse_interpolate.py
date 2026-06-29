# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""CUDA graph tests for sparse articulation coarse interpolation."""

import unittest

import numpy as np
import warp as wp

from newton._src.solvers.phoenx.articulations.coarse_interpolate import CoarseInterpolationSolver
from newton._src.solvers.phoenx.articulations.device import ArticulationDeviceSystem
from newton._src.solvers.phoenx.articulations.symbolic import compute_block_sparse_symbolic
from newton._src.solvers.phoenx.constraints.constraint_joint import JOINT_MODE_BALL_SOCKET
from newton._src.solvers.phoenx.tests.test_articulation_dvi import _make_adbs_world


class TestPhoenXCoarseInterpolation(unittest.TestCase):
    def test_cycle_operator_and_momentum_under_cuda_graph(self):
        device = wp.get_preferred_device()
        if not device.is_cuda:
            self.skipTest("coarse interpolation tests require CUDA graph capture")

        blocks = 8
        dynamic_bodies = np.arange(1, blocks + 1, dtype=np.int32)
        body1 = dynamic_bodies
        body2 = np.roll(dynamic_bodies, -1)
        angle = 2.0 * np.pi * np.arange(blocks) / blocks
        positions = np.zeros((blocks + 1, 3), dtype=np.float32)
        positions[1:, 0] = np.cos(angle)
        positions[1:, 1] = np.sin(angle)
        world = _make_adbs_world(
            device,
            body1,
            body2,
            np.full(blocks, int(JOINT_MODE_BALL_SOCKET), dtype=np.int32),
            positions_np=positions,
            world_kwargs={"cache_articulation_topology": True, "gravity": (0.0, 0.0, 0.0)},
        )
        topology = world.articulation_topology
        symbolic = compute_block_sparse_symbolic(
            topology.active_body1,
            topology.active_body2,
            topology.active_row_counts,
            use_meca=False,
        )
        system = ArticulationDeviceSystem.from_topology(topology, device, symbolic)
        coarse_blocks = blocks // 2
        interpolation: list[tuple[tuple[int, float], ...]] = []
        for fine in range(blocks):
            if fine % 2 == 0:
                interpolation.append(((fine // 2, 1.0),))
            else:
                interpolation.append(((fine // 2, 0.5), (((fine + 1) % blocks) // 2, 0.5)))
        coarse = CoarseInterpolationSolver(
            interpolation,
            symbolic.n_off_row_idx[: symbolic.nnz_n],
            symbolic.n_off_col_idx[: symbolic.nnz_n],
            rows=3,
            color_sweeps=16,
            device=device,
        )

        rng = np.random.default_rng(83)
        velocity = np.zeros_like(positions)
        angular_velocity = np.zeros_like(positions)
        velocity[1:] = rng.normal(size=velocity[1:].shape)
        angular_velocity[1:] = rng.normal(size=angular_velocity[1:].shape)
        world.bodies.velocity.assign(velocity)
        world.bodies.angular_velocity.assign(angular_velocity)
        linear_before = velocity[1:].sum(axis=0)
        angular_before = (angular_velocity[1:] + np.cross(positions[1:], velocity[1:])).sum(axis=0)
        with wp.ScopedCapture(device=device) as capture:
            system.populate_from_adbs_constraints(world.constraints, world.bodies, dt=1.0e-4, device=device)
            system.compute_residual(world.bodies, dt=1.0e-4, recovery_speed=0.0, device=device)
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

        velocity_after = world.bodies.velocity.numpy()[1:]
        angular_velocity_after = world.bodies.angular_velocity.numpy()[1:]
        linear_after = velocity_after.sum(axis=0)
        angular_after = (angular_velocity_after + np.cross(positions[1:], velocity_after)).sum(axis=0)
        np.testing.assert_allclose(linear_after, linear_before, rtol=3.0e-6, atol=3.0e-6)
        np.testing.assert_allclose(angular_after, angular_before, rtol=3.0e-6, atol=3.0e-6)

        rows = 3
        fine_matrix = np.zeros((blocks * rows, blocks * rows), dtype=np.float64)
        fine_diag = system.block_diag.numpy()[:blocks, :rows, :rows]
        fine_off = system.block_off.numpy()[: symbolic.nnz_n, :rows, :rows]
        for block in range(blocks):
            row = slice(block * rows, (block + 1) * rows)
            fine_matrix[row, row] = fine_diag[block]
        for edge, (row_block, col_block) in enumerate(
            zip(symbolic.n_off_row_idx[: symbolic.nnz_n], symbolic.n_off_col_idx[: symbolic.nnz_n], strict=True)
        ):
            row = slice(row_block * rows, (row_block + 1) * rows)
            col = slice(col_block * rows, (col_block + 1) * rows)
            fine_matrix[row, col] = fine_off[edge]
            fine_matrix[col, row] = fine_off[edge].T
        prolongation = np.zeros((blocks * rows, coarse_blocks * rows))
        for fine, entries in enumerate(interpolation):
            for coarse_block, weight in entries:
                for row in range(rows):
                    prolongation[fine * rows + row, coarse_block * rows + row] = weight
        expected = prolongation.T @ fine_matrix @ prolongation
        actual = np.zeros_like(expected)
        coarse_diag = coarse.diag.numpy()[:coarse_blocks, :rows, :rows]
        coarse_off = coarse.off.numpy()[: len(coarse.coarse_pairs), :rows, :rows]
        for block in range(coarse_blocks):
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
