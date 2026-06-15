# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for PhoenX full-coordinate articulation DVI pieces."""

from __future__ import annotations

import unittest

import numpy as np
import warp as wp

from newton._src.solvers.phoenx.articulations import (
    ArticulationDeviceSystem,
    ArticulationTopology,
    PrefactorizedArticulationSystem,
    compute_block_sparse_symbolic,
    d6_constraint_row_count,
    factorize_ldlt,
    joint_constraint_row_count,
    revolute_rows,
)
from newton._src.solvers.phoenx.body import MOTION_DYNAMIC, MOTION_STATIC, body_container_zeros
from newton._src.solvers.phoenx.constraints.constraint_joint import (
    DRIVE_MODE_OFF,
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


def _make_adbs_world(
    device,
    body1_np: np.ndarray,
    body2_np: np.ndarray,
    mode_np: np.ndarray,
    *,
    positions_np: np.ndarray | None = None,
    world_kwargs: dict | None = None,
) -> PhoenXWorld:
    body_count = int(max(body1_np.max(initial=0), body2_np.max(initial=0)) + 1)
    joint_count = int(mode_np.size)
    bodies = body_container_zeros(body_count, device=device)

    if positions_np is None:
        positions_np = np.zeros((body_count, 3), dtype=np.float32)
    bodies.position.assign(positions_np.astype(np.float32))

    orientations_np = np.zeros((body_count, 4), dtype=np.float32)
    orientations_np[:, 3] = 1.0
    bodies.orientation.assign(orientations_np)

    inverse_mass_np = np.ones(body_count, dtype=np.float32)
    inverse_mass_np[0] = 0.0
    bodies.inverse_mass.assign(inverse_mass_np)
    motion_type_np = np.full(body_count, int(MOTION_DYNAMIC), dtype=np.int32)
    motion_type_np[0] = int(MOTION_STATIC)
    bodies.motion_type.assign(motion_type_np)

    inverse_inertia_np = np.repeat(np.eye(3, dtype=np.float32)[None, :, :], body_count, axis=0)
    inverse_inertia_np[0] = 0.0
    bodies.inverse_inertia_world.assign(inverse_inertia_np)

    constraints = PhoenXWorld.make_constraint_container(num_joints=joint_count, device=device)
    if world_kwargs is None:
        world_kwargs = {}
    world = PhoenXWorld(
        bodies=bodies,
        constraints=constraints,
        num_joints=joint_count,
        num_worlds=1,
        device=device,
        **world_kwargs,
    )

    anchor1_np = np.zeros((joint_count, 3), dtype=np.float32)
    anchor2_np = np.tile(np.array([0.0, 0.0, 1.0], dtype=np.float32), (joint_count, 1))
    zeros_np = np.zeros(joint_count, dtype=np.float32)
    mode_drive_np = np.full(joint_count, int(DRIVE_MODE_OFF), dtype=np.int32)
    min_np = np.ones(joint_count, dtype=np.float32)
    max_np = -np.ones(joint_count, dtype=np.float32)

    world.initialize_actuated_double_ball_socket_joints(
        wp.array(body1_np.astype(np.int32), dtype=wp.int32, device=device),
        wp.array(body2_np.astype(np.int32), dtype=wp.int32, device=device),
        wp.array(anchor1_np, dtype=wp.vec3f, device=device),
        wp.array(anchor2_np, dtype=wp.vec3f, device=device),
        wp.array(zeros_np, dtype=wp.float32, device=device),
        wp.array(zeros_np, dtype=wp.float32, device=device),
        wp.array(mode_np.astype(np.int32), dtype=wp.int32, device=device),
        wp.array(mode_drive_np, dtype=wp.int32, device=device),
        wp.array(zeros_np, dtype=wp.float32, device=device),
        wp.array(zeros_np, dtype=wp.float32, device=device),
        wp.array(zeros_np, dtype=wp.float32, device=device),
        wp.array(zeros_np, dtype=wp.float32, device=device),
        wp.array(zeros_np, dtype=wp.float32, device=device),
        wp.array(min_np, dtype=wp.float32, device=device),
        wp.array(max_np, dtype=wp.float32, device=device),
        wp.array(zeros_np, dtype=wp.float32, device=device),
        wp.array(zeros_np, dtype=wp.float32, device=device),
        wp.array(zeros_np, dtype=wp.float32, device=device),
        wp.array(zeros_np, dtype=wp.float32, device=device),
    )
    return world


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

        device = wp.get_preferred_device()
        device_system = ArticulationDeviceSystem.from_topology(topology, device)
        device_system.jacobian.assign(jac.astype(np.float32))
        inv_mass_wp = wp.array(inv_mass.astype(np.float32), dtype=wp.float32, device=device)
        inv_inertia_wp = wp.array(inv_inertia.astype(np.float32), dtype=wp.mat33f, device=device)
        device_system.assemble_dense_matrix(inv_mass_wp, inv_inertia_wp, device=device)
        np.testing.assert_allclose(device_system.matrix.numpy()[:6, :6], expected, rtol=1.0e-6, atol=1.0e-6)

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

    def test_device_populates_rows_from_initialized_adbs_constraints(self):
        device = wp.get_preferred_device()
        body1 = np.array([0, 1, 2, 3, 4], dtype=np.int32)
        body2 = np.array([1, 2, 3, 4, 5], dtype=np.int32)
        mode = np.array(
            [
                int(JOINT_MODE_BALL_SOCKET),
                int(JOINT_MODE_REVOLUTE),
                int(JOINT_MODE_PRISMATIC),
                int(JOINT_MODE_FIXED),
                int(JOINT_MODE_CYLINDRICAL),
            ],
            dtype=np.int32,
        )
        world = _make_adbs_world(device, body1, body2, mode)

        device_system = world.articulation_device_system
        self.assertIsNotNone(device_system)
        device_system.populate_from_adbs_constraints(world.constraints, world.bodies, device=device)

        jac = device_system.jacobian.numpy()[: device_system.total_rows]
        violation = device_system.violation.numpy()[: device_system.total_rows]

        self.assertEqual(device_system.total_rows, 23)
        np.testing.assert_allclose(violation, np.zeros(23, dtype=np.float32), atol=1.0e-6)
        np.testing.assert_allclose(jac[:3, :3], -np.eye(3, dtype=np.float32), atol=1.0e-6)
        np.testing.assert_allclose(jac[:3, 6:9], np.eye(3, dtype=np.float32), atol=1.0e-6)

        offsets = world.articulation_topology.active_block_offsets
        revolute_angular = jac[offsets[1] + 3 : offsets[1] + 5]
        np.testing.assert_allclose(revolute_angular[:, :3], 0.0, atol=1.0e-6)
        np.testing.assert_allclose(revolute_angular[:, 6:9], 0.0, atol=1.0e-6)
        np.testing.assert_allclose(revolute_angular[:, 9:12], -revolute_angular[:, 3:6], atol=1.0e-6)
        np.testing.assert_allclose(np.linalg.norm(revolute_angular[:, 3:6], axis=1), np.ones(2), atol=1.0e-6)

        prismatic_offset = offsets[2]
        np.testing.assert_allclose(
            np.linalg.norm(jac[prismatic_offset : prismatic_offset + 2, :3], axis=1),
            np.ones(2),
            atol=1.0e-6,
        )
        np.testing.assert_allclose(jac[prismatic_offset + 4, :3], 0.0, atol=1.0e-6)
        np.testing.assert_allclose(jac[prismatic_offset + 4, 6:9], 0.0, atol=1.0e-6)
        np.testing.assert_allclose(jac[prismatic_offset + 4, 9:12], -jac[prismatic_offset + 4, 3:6], atol=1.0e-6)

        device_system.assemble_dense_matrix(
            world.bodies.inverse_mass, world.bodies.inverse_inertia_world, device=device
        )
        matrix = device_system.matrix.numpy()[: device_system.total_rows, : device_system.total_rows]
        np.testing.assert_allclose(matrix, matrix.T, atol=1.0e-6)
        np.testing.assert_allclose(matrix[:3, :3], np.eye(3, dtype=np.float32), atol=1.0e-6)

    def test_device_population_updates_violations_after_body_motion(self):
        device = wp.get_preferred_device()
        world = _make_adbs_world(
            device,
            np.array([0], dtype=np.int32),
            np.array([1], dtype=np.int32),
            np.array([int(JOINT_MODE_BALL_SOCKET)], dtype=np.int32),
        )
        moved_positions = np.array([[0.0, 0.0, 0.0], [0.25, -0.5, 0.75]], dtype=np.float32)
        world.bodies.position.assign(moved_positions)

        device_system = world.articulation_device_system
        self.assertIsNotNone(device_system)
        device_system.populate_from_adbs_constraints(world.constraints, world.bodies, device=device)

        np.testing.assert_allclose(
            device_system.violation.numpy()[:3],
            np.array([0.25, -0.5, 0.75], dtype=np.float32),
            atol=1.0e-6,
        )

    def test_host_dvi_solve_applies_articulation_impulse(self):
        device = wp.get_preferred_device()
        world = _make_adbs_world(
            device,
            np.array([0], dtype=np.int32),
            np.array([1], dtype=np.int32),
            np.array([int(JOINT_MODE_BALL_SOCKET)], dtype=np.int32),
        )
        moved_positions = np.array([[0.0, 0.0, 0.0], [0.2, -0.1, 0.05]], dtype=np.float32)
        world.bodies.position.assign(moved_positions)

        solved = world.solve_articulations_dvi_host(dt=0.1, alpha=0.0)

        self.assertTrue(solved)
        expected_velocity = np.array([-2.0, 1.0, -0.5], dtype=np.float32)
        np.testing.assert_allclose(world.bodies.velocity.numpy()[1], expected_velocity, rtol=1.0e-5, atol=1.0e-5)
        np.testing.assert_allclose(world.bodies.angular_velocity.numpy()[1], np.zeros(3, dtype=np.float32), atol=1.0e-6)

    def test_step_runs_opt_in_host_dvi_as_joint_owner(self):
        device = wp.get_preferred_device()
        world = _make_adbs_world(
            device,
            np.array([0], dtype=np.int32),
            np.array([1], dtype=np.int32),
            np.array([int(JOINT_MODE_BALL_SOCKET)], dtype=np.int32),
            world_kwargs={"articulation_dvi_host": True, "velocity_iterations": 0, "gravity": (0.0, 0.0, 0.0)},
        )
        world.bodies.position.assign(np.array([[0.0, 0.0, 0.0], [0.2, -0.1, 0.05]], dtype=np.float32))

        self.assertTrue(world.articulation_dvi_replaces_joint_pgs)
        self.assertTrue(world._dispatch_specialization_flags()["skip_joint_pgs"])

        world.step(0.1)

        expected_velocity = np.array([-2.0, 1.0, -0.5], dtype=np.float32)
        np.testing.assert_allclose(world.bodies.position.numpy()[1], np.zeros(3, dtype=np.float32), atol=1.0e-5)
        np.testing.assert_allclose(world.bodies.velocity.numpy()[1], expected_velocity, rtol=1.0e-5, atol=1.0e-5)

    def test_host_dvi_joint_owner_other_dispatchers(self):
        device = wp.get_preferred_device()
        dispatcher_kwargs = (
            {"step_layout": "single_world"},
            {"multi_world_scheduler": "block_world"},
        )
        for extra_kwargs in dispatcher_kwargs:
            with self.subTest(extra_kwargs=extra_kwargs):
                world_kwargs = {
                    "articulation_dvi_host": True,
                    "velocity_iterations": 0,
                    "gravity": (0.0, 0.0, 0.0),
                }
                world_kwargs.update(extra_kwargs)
                world = _make_adbs_world(
                    device,
                    np.array([0], dtype=np.int32),
                    np.array([1], dtype=np.int32),
                    np.array([int(JOINT_MODE_BALL_SOCKET)], dtype=np.int32),
                    world_kwargs=world_kwargs,
                )
                world.bodies.position.assign(np.array([[0.0, 0.0, 0.0], [0.2, -0.1, 0.05]], dtype=np.float32))

                world.step(0.1)

                np.testing.assert_allclose(world.bodies.position.numpy()[1], np.zeros(3, dtype=np.float32), atol=1.0e-5)

    def test_host_dvi_can_run_as_post_pgs_correction(self):
        device = wp.get_preferred_device()
        world = _make_adbs_world(
            device,
            np.array([0], dtype=np.int32),
            np.array([1], dtype=np.int32),
            np.array([int(JOINT_MODE_BALL_SOCKET)], dtype=np.int32),
            world_kwargs={
                "articulation_dvi_host": True,
                "articulation_dvi_replaces_joint_pgs": False,
                "velocity_iterations": 0,
                "gravity": (0.0, 0.0, 0.0),
            },
        )

        self.assertFalse(world.articulation_dvi_replaces_joint_pgs)
        self.assertFalse(world._dispatch_specialization_flags()["skip_joint_pgs"])

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
        self.assertIsNotNone(world.articulation_device_system)
        np.testing.assert_array_equal(world.articulation_topology.body1, np.array([-1, 1], dtype=np.int32))
        self.assertEqual(world.articulation_topology.total_rows, 8)
        self.assertEqual(world.articulation_system.symbolic.total_rows, 8)


if __name__ == "__main__":
    unittest.main()
