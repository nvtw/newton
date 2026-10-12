# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import math
import unittest
from types import SimpleNamespace

import numpy as np
import warp as wp

import newton
from newton.examples.phoenx.example_phoenx_ragdoll_pile import Example, build_scene
from newton.examples.phoenx.ragdoll_pile_scene import BODIES, DEFAULT_GRID, ragdoll_positions, scene_bounds
from newton.viewer import ViewerNull


class TestPhoenXRagdollPile(unittest.TestCase):
    def test_reference_size(self):
        count = math.prod(DEFAULT_GRID)
        self.assertEqual(count, 32_000)
        self.assertEqual(6 * count + 1, 192_001)
        self.assertEqual(5 * count, 160_000)
        self.assertEqual(sum(1 for _ in ragdoll_positions()), count)

    def test_shared_world_and_spherical_d6_topology(self):
        builder = build_scene(grid=(2, 3, 2))
        self.assertEqual(builder.body_count, 73)
        self.assertEqual(builder.shape_count, 73)
        self.assertEqual(builder.joint_count, 60)
        self.assertEqual(set(builder.body_world), {-1})
        self.assertEqual(set(builder.joint_type), {newton.JointType.D6})
        self.assertEqual(set(builder.joint_dof_dim), {(0, 3)})
        self.assertEqual(builder.body_mass[0], 0.0)
        self.assertTrue(all(mass > 0 for mass in builder.body_mass[1:]))
        for doll in range(12):
            start = 1 + 6 * doll
            self.assertEqual(builder.shape_collision_group[start : start + 6], [-(doll + 1)] * 6)
            for joint in range(5 * doll, 5 * (doll + 1)):
                self.assertEqual(builder.joint_parent[joint], start)
                parent = np.asarray(builder.body_q[start][:3]) + np.asarray(builder.joint_X_p[joint][:3])
                child = np.asarray(builder.body_q[builder.joint_child[joint]][:3])
                child += np.asarray(builder.joint_X_c[joint][:3])
                np.testing.assert_allclose(parent, child, atol=1.0e-6)

    def test_initial_boxes_are_clear_and_seeded(self):
        grid = (2, 2, 2)
        first = build_scene(grid=grid, seed=7)
        second = build_scene(grid=grid, seed=7)
        np.testing.assert_array_equal(first.shape_color, second.shape_color)
        centers = np.asarray(first.body_q)[1:, :3]
        half_extents = np.tile(np.asarray([body.half_extents for body in BODIES]), (8, 1))
        overlaps = np.all(
            np.abs(centers[:, None] - centers[None, :]) < half_extents[:, None] + half_extents[None, :], axis=-1
        )
        np.fill_diagonal(overlaps, False)
        self.assertFalse(overlaps.any())
        self.assertGreater(float((centers - half_extents)[:, 2].min()), 0.0)
        lower, upper = scene_bounds(grid)
        np.testing.assert_allclose((centers - half_extents).min(axis=0), lower, atol=1.0e-6)
        np.testing.assert_allclose((centers + half_extents).max(axis=0), upper, atol=1.0e-6)

    def test_invalid_grid(self):
        for grid in ((0, 1, 1), (1, -1, 1), (1, 1)):
            with self.subTest(grid=grid), self.assertRaises(ValueError):
                build_scene(grid=grid)

    @unittest.skipUnless(wp.is_cuda_available(), "The colored rigid solver requires CUDA")
    def test_single_drop_predicts_floor_contact(self):
        """Prevent impact penetration while retaining the authored 10 mm gap."""
        self._check_single_drop("exact", 4)

    @unittest.skipUnless(wp.is_cuda_available(), "The colored rigid solver requires CUDA")
    def test_colored_drop_predicts_floor_contact(self):
        """Retain predictive contact coverage for the iterative joint path."""
        self._check_single_drop("colored", 16)

    def _check_single_drop(self, joint_solver, iterations):
        viewer = ViewerNull()
        try:
            example = Example(
                viewer,
                SimpleNamespace(grid=(1, 1, 1), seed=42, substeps=8, iterations=iterations, joint_solver=joint_solver),
            )
            self.assertEqual(example.joint_solver, joint_solver)
            self.assertEqual(example.solver.world.constraints.bilateral.enabled, joint_solver == "colored")
            self.assertTrue(example.solver.world._singleworld_overflow_only_mass_splitting)
            poses = example.state.body_q.numpy()
            poses[1:, 2] += 3.0
            example.state.body_q.assign(poses)
            half_extents = np.asarray([body.half_extents for body in BODIES])
            minimum = 0.0
            saw_contact = False
            for _ in range(120):
                example.step()
                poses = example.state.body_q.numpy()
                x, y, z, w = poses[1:, 3:].T
                rotation_z = np.stack((2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)), axis=1)
                bottom = poses[1:, 2] - np.sum(np.abs(rotation_z) * half_extents, axis=1)
                minimum = min(minimum, float(bottom.min()))
                count = int(example.contacts.rigid_contact_count.numpy()[0])
                if count:
                    saw_contact = True
                    first = example.contacts.rigid_contact_shape0.numpy()[:count]
                    second = example.contacts.rigid_contact_shape1.numpy()[:count]
                    self.assertTrue(np.all((first == 0) | (second == 0)), "Unexpected ragdoll self-contact")
            np.testing.assert_allclose(example.model.shape_gap.numpy()[1:], 0.01)
            self.assertTrue(saw_contact)
            self.assertGreater(minimum, -0.002, f"Floor penetration reached {-minimum * 1000:.3f} mm")
            example.test_final()
        finally:
            viewer.close()


if __name__ == "__main__":
    unittest.main()
