# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""History-only patch lifecycle: no friction forces, pose repair or scene assumptions."""

import unittest

import numpy as np
import warp as wp

from newton._src.solvers.phoenx.constraints.friction_patch_history import (
    allocate_friction_patch_history,
    check_friction_patch_history,
    update_friction_patch_history,
)


class Inputs:
    """Preallocated external CSR/contact inputs, overwritten between generations."""

    def __init__(self, device, patches=4, points=8):
        self.device = device
        self.patches, self.points = patches, points
        self.state = allocate_friction_patch_history(patches, points, device)
        self.counts = wp.zeros(2, dtype=wp.int32, device=device)
        self.offsets = wp.zeros(patches + 1, dtype=wp.int32, device=device)
        self.members = wp.zeros(points, dtype=wp.int32, device=device)
        self.keys = wp.zeros(points, dtype=wp.vec4i, device=device)
        self.normals = wp.zeros(points, dtype=wp.vec3f, device=device)
        self.matches = wp.zeros(points, dtype=wp.int32, device=device)
        self.birth = wp.zeros((14, patches), dtype=wp.float32, device=device)

    def load(self, groups, keys, *, matches=None, normals=None, seed=10):
        n = len(keys)
        offsets = np.zeros(self.patches + 1, np.int32)
        offsets[: len(groups) + 1] = np.cumsum([0, *map(len, groups)])
        members = np.zeros(self.points, np.int32)
        members[:n] = [point for group in groups for point in group]
        key_array = np.zeros((self.points, 4), np.int32)
        key_array[:n] = np.asarray(keys, dtype=np.int32).reshape(n, 4)
        normal_array = np.zeros((self.points, 3), np.float32)
        normal_array[:n] = normals if normals is not None else [0, 0, 1]
        match_array = np.full(self.points, -1, np.int32)
        if matches is not None:
            match_array[:n] = matches
        self.counts.assign(np.array([len(groups), n], np.int32))
        self.offsets.assign(offsets)
        self.members.assign(members)
        self.keys.assign(key_array)
        self.normals.assign(normal_array)
        self.matches.assign(match_array)
        self.birth.assign(np.arange(14 * self.patches, dtype=np.float32).reshape(14, self.patches) + seed)

    def update(self):
        update_friction_patch_history(
            self.state,
            self.counts,
            self.offsets,
            self.members,
            self.keys,
            self.normals,
            self.matches,
            self.birth,
            normal_cosine=0.99,
            device=self.device,
        )

    def checked(self):
        self.update()
        check_friction_patch_history(self.state)


class TestFrictionPatchHistoryCPU(unittest.TestCase):
    """All tests are input/lifecycle checks independent of the contact solver."""

    device = "cpu"

    def test_reorder_arbitrary_pairs_and_worlds(self):
        f = Inputs(self.device)
        a, b = (3, 71, 103, 5), (4, 71, 103, 6)
        keys = [a, a, b, b]
        f.load([[0, 1], [2, 3]], keys)
        f.checked()
        old = f.state.birth.numpy().copy()
        f.load([[2, 3], [0, 1]], keys, matches=[0, 1, 2, 3], seed=200)
        f.checked()
        np.testing.assert_array_equal(f.state.birth.numpy()[:, :2], old[:, [1, 0]])
        np.testing.assert_array_equal(f.state.point_patch.numpy()[:4], [1, 1, 0, 0])
        np.testing.assert_array_equal(f.state.age.numpy()[:2], [1, 1])

    def test_disjoint_patches_same_key_do_not_alias(self):
        f = Inputs(self.device)
        key = (8, 71, 103, 5)
        f.load([[0, 1], [2, 3]], [key] * 4)
        f.checked()
        old = f.state.birth.numpy().copy()
        f.load([[2, 3], [0, 1]], [key] * 4, matches=[0, 1, 2, 3], seed=200)
        f.checked()
        np.testing.assert_array_equal(f.state.birth.numpy()[:, :2], old[:, [1, 0]])
        # Same compatibility key without material-point lineage is not identity.
        f.load([[0, 1], [2, 3]], [key] * 4, matches=[-1] * 4, seed=300)
        f.checked()
        np.testing.assert_array_equal(f.state.birth.numpy()[:, :2], f.birth.numpy()[:, :2])
        # Neither child may clone history from one predecessor.
        f.load([[0, 1], [2, 3]], [key] * 4, matches=[0, 1, 0, 1], seed=400)
        f.checked()
        np.testing.assert_array_equal(f.state.birth.numpy()[:, :2], f.birth.numpy()[:, :2])

    def test_new_point_does_not_inherit_old_birth(self):
        f = Inputs(self.device)
        key = (2, 91, 8, 17)
        f.load([[0, 1]], [key, key])
        f.checked()
        f.load([[0, 1]], [key, key], matches=[0, -1], seed=500)
        f.checked()
        np.testing.assert_array_equal(f.state.birth.numpy()[:, 0], f.birth.numpy()[:, 0])
        self.assertEqual(int(f.state.age.numpy()[0]), 0)

    def test_split_merge_and_duplicate_matches_rebirth(self):
        f = Inputs(self.device)
        key = (0, 9, 8, 3)
        f.load([[0, 1]], [key, key])
        f.checked()
        f.load([[0], [1]], [key, key], matches=[0, 1], seed=20)
        f.checked()
        np.testing.assert_array_equal(f.state.age.numpy()[:2], [0, 0])
        f.load([[0, 1]], [key, key], matches=[0, 1], seed=30)
        f.checked()
        self.assertEqual(int(f.state.age.numpy()[0]), 0)
        f.load([[0, 1]], [key, key], matches=[0, 0], seed=40)
        f.checked()
        np.testing.assert_array_equal(f.state.birth.numpy()[:, 0], f.birth.numpy()[:, 0])

    def test_material_normal_and_explicit_break(self):
        f = Inputs(self.device)
        key = (0, 9, 8, 3)
        f.load([[0]], [key])
        f.checked()
        f.load([[0]], [(0, 9, 8, 4)], matches=[0], seed=20)
        f.checked()
        self.assertEqual(int(f.state.age.numpy()[0]), 0)
        f.load([[0]], [(0, 9, 8, 4)], matches=[0], normals=[[1, 0, 0]], seed=30)
        f.checked()
        self.assertEqual(int(f.state.age.numpy()[0]), 0)
        f.state.broken.fill_(1)
        f.load([[0]], [(0, 9, 8, 4)], matches=[0], normals=[[1, 0, 0]], seed=40)
        f.checked()
        np.testing.assert_array_equal(f.state.birth.numpy()[:, 0], f.birth.numpy()[:, 0])

    def test_incompatible_group_rejected_without_overwriting(self):
        for keys, normals in (
            ([(0, 9, 8, 3), (1, 9, 8, 3)], None),
            ([(0, 9, 8, 3), (0, 9, 8, 4)], None),
            ([(0, 9, 8, 3), (0, 9, 8, 3)], [[0, 0, 1], [0, 0, -1]]),
        ):
            with self.subTest(keys=keys, normals=normals):
                f = Inputs(self.device)
                f.load([[0, 1]], [(0, 9, 8, 3)] * 2)
                f.checked()
                before = f.state.birth.numpy().copy()
                f.load([[0, 1]], keys, normals=normals, seed=99)
                f.update()
                with self.assertRaises(ValueError):
                    check_friction_patch_history(f.state)
                np.testing.assert_array_equal(before, f.state.birth.numpy())

    def test_nonunit_normal_rejected_without_overwriting(self):
        f = Inputs(self.device)
        key = (0, 9, 8, 3)
        f.load([[0]], [key])
        f.checked()
        before = f.state.birth.numpy().copy()
        f.load([[0]], [key], normals=[[0, 0, 2]], matches=[0], seed=99)
        f.update()
        with self.assertRaises(ValueError):
            check_friction_patch_history(f.state)
        np.testing.assert_array_equal(before, f.state.birth.numpy())

    def test_nonfinite_birth_rejected_without_overwriting(self):
        f = Inputs(self.device)
        key = (0, 9, 8, 3)
        f.load([[0]], [key])
        f.checked()
        before = f.state.birth.numpy().copy()
        f.birth.fill_(float("nan"))
        f.update()
        with self.assertRaises(ValueError):
            check_friction_patch_history(f.state)
        np.testing.assert_array_equal(before, f.state.birth.numpy())

    def test_capacity_and_duplicate_members_preserve_live_state(self):
        f = Inputs(self.device)
        key = (0, 9, 8, 3)
        f.load([[0, 1]], [key, key])
        f.checked()
        before = f.state.birth.numpy().copy()
        f.counts.assign(np.array([f.patches + 1, 2], np.int32))
        f.update()
        with self.assertRaises(ValueError):
            check_friction_patch_history(f.state)
        np.testing.assert_array_equal(before, f.state.birth.numpy())
        f.load([[0, 0]], [key, key])
        f.update()
        with self.assertRaises(ValueError):
            check_friction_patch_history(f.state)
        np.testing.assert_array_equal(before, f.state.birth.numpy())
        self.assertEqual(int(f.state.count.numpy()[0]), 1)

    def test_arbitrary_count_and_empty_reuse(self):
        f = Inputs(self.device, points=150)
        key = (7, 44, 53, 89)
        f.load([list(range(129))], [key] * 129)
        f.checked()
        f.load([list(range(129))], [key] * 129, matches=list(range(129)), seed=30)
        f.checked()
        self.assertEqual(int(f.state.age.numpy()[0]), 1)
        f.load([], [])
        f.checked()
        self.assertEqual(int(f.state.count.numpy()[0]), 0)
        f.load([[0]], [key], matches=[0], seed=40)
        f.checked()
        self.assertEqual(int(f.state.age.numpy()[0]), 0)
        np.testing.assert_array_equal(f.state.birth.numpy()[:, 0], f.birth.numpy()[:, 0])


class TestFrictionPatchHistoryCUDA(TestFrictionPatchHistoryCPU):
    """Same lifecycle gates plus true repeated dynamic-count graph execution."""

    device = "cuda:0"

    @classmethod
    def setUpClass(cls):
        if not wp.is_cuda_available():
            raise unittest.SkipTest("CUDA is required")

    def test_capture_dynamic_empty_reuse_and_overflow(self):
        f = Inputs(self.device)
        key = (1, 11, 31, 8)
        f.load([[0, 1]], [key, key])
        f.checked()
        with wp.ScopedCapture(device=self.device) as capture:
            f.update()
        for groups, keys, matches, seed in (
            ([], [], [], 20),
            ([[0, 1]], [key, key], [-1, -1], 30),
            ([[0, 1]], [key, key], [0, 1], 40),
        ):
            f.load(groups, keys, matches=matches, seed=seed)
            wp.capture_launch(capture.graph)
            check_friction_patch_history(f.state)
        self.assertEqual(int(f.state.age.numpy()[0]), 1)
        expected = np.arange(14 * f.patches, dtype=np.float32).reshape(14, f.patches)[:, 0] + 30
        np.testing.assert_array_equal(f.state.birth.numpy()[:, 0], expected)
        before = f.state.birth.numpy().copy()
        f.counts.assign(np.array([99, 99], np.int32))
        wp.capture_launch(capture.graph)
        with self.assertRaises(ValueError):
            check_friction_patch_history(f.state)
        np.testing.assert_array_equal(before, f.state.birth.numpy())


if __name__ == "__main__":
    unittest.main()
