# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Persistent anchors must follow geometry and material, not contact indices."""

import unittest

import numpy as np
import warp as wp

from newton._src.solvers.phoenx.constraints.contact_tgs_anchors import allocate, prepare
from newton._src.solvers.phoenx.constraints.contact_tgs_partition import allocate as allocate_partition
from newton._src.solvers.phoenx.constraints.contact_tgs_partition import partition


class TestNormalPatchAnchors(unittest.TestCase):
    def test_permutation_material_break_and_separation(self):
        """Retain compatible anchors and reject broken or separated history."""
        for device in ["cpu"] + (["cuda:0"] if wp.is_cuda_available() else []):
            with self.subTest(device=device):
                self._check_device(device)

    def _check_device(self, device):
        first = wp.array([0, 4], dtype=int, device=device)
        count = wp.array([4, 2], dtype=int, device=device)
        active = wp.array([2], dtype=int, device=device)
        absent = wp.array([0], dtype=int, device=device)
        normals = np.array([[0, 0, 1], [1, 0, 0], [0, 0, 1], [1, 0, 0], [0, 0, 1], [0, 0, 1]], dtype=np.float32)
        points = np.array(
            [[-0.1, 0, 0], [0, -0.1, 0], [0.1, 0, 0], [0, 0.1, 0], [1, 0, 0], [1.2, 0, 0]], dtype=np.float32
        )
        keys = np.array([[0, 0, 1, 7], [0, 0, 1, 8]], dtype=np.int32)
        wk = wp.array(keys, dtype=wp.vec4i, device=device)
        poses_np = np.array([[0, 0, 0, 0, 0, 0, 1], [0.03, 0.04, 0.05, 0, 0, 0, 1]], dtype=np.float32)
        poses = wp.array(poses_np, dtype=wp.transformf, device=device)
        gaps = wp.zeros(6, dtype=float, device=device)
        old_part, new_part = allocate_partition(6, 2, device), allocate_partition(6, 2, device)
        empty = allocate_partition(6, 2, device)
        old, new, blank = allocate(6, device), allocate(6, device), allocate(6, device)
        wn = wp.array(normals, dtype=wp.vec3f, device=device)
        wpnts = wp.array(points, dtype=wp.vec3f, device=device)
        wp.launch(partition, 2, [wn, first, count, active, 0.999, old_part], device=device)
        wp.launch(
            prepare,
            2,
            [old_part, empty, wk, wk, active, absent, poses, wpnts, gaps, 0.00025, 0.0004, blank, old],
            device=device,
        )
        np.testing.assert_array_equal(old.count.numpy()[[0, 1, 4]], [2, 2, 2])
        original0 = old.local0.numpy().copy()
        original1 = old.local1.numpy().copy()
        order = [1, 0, 3, 2, 5, 4]
        wn.assign(normals[order])
        shifted = points[order] + [0, 0.00001, 0]
        wpnts.assign(shifted.astype(np.float32))
        wp.launch(partition, 2, [wn, first, count, active, 0.999, new_part], device=device)

        def update(current_keys=wk):
            wp.launch(
                prepare,
                2,
                [
                    new_part,
                    old_part,
                    current_keys,
                    wk,
                    active,
                    active,
                    poses,
                    wpnts,
                    gaps,
                    0.00025,
                    0.0004,
                    old,
                    new,
                ],
                device=device,
            )

        update()
        np.testing.assert_array_equal(new.source.numpy()[[0, 1, 4]], [1, 0, 4])
        np.testing.assert_array_equal(new.local0.numpy()[[0, 1, 4]], original0[[1, 0, 4]])
        np.testing.assert_array_equal(new.local1.numpy()[[0, 1, 4]], original1[[1, 0, 4]])
        changed = keys.copy()
        changed[1, 3] = 9
        update(wp.array(changed, dtype=wp.vec4i, device=device))
        self.assertEqual(int(new.source.numpy()[4]), -1)
        self.assertGreater(float(np.max(np.abs(new.local0.numpy()[4] - original0[4]))), 1e-6)
        broken = np.zeros(6, dtype=np.int32)
        broken[1] = 1
        old.broken.assign(broken)
        update()
        self.assertEqual(int(new.source.numpy()[0]), -1)
        self.assertEqual(int(new.source.numpy()[1]), 0)
        old.broken.zero_()
        moved = poses_np.copy()
        moved[1, 2] += 0.001
        poses.assign(moved)
        update()
        self.assertEqual(int(new.source.numpy()[1]), -1)
        self.assertEqual(int(new.source.numpy()[0]), 1)
        # Speculative points beyond the friction threshold cannot
        # create new anchors after the previous history disappears.
        gaps.assign(np.full(6, 0.001, dtype=np.float32))
        wp.launch(
            prepare,
            2,
            [new_part, empty, wk, wk, active, absent, poses, wpnts, gaps, 0.00025, 0.0004, blank, new],
            device=device,
        )
        np.testing.assert_array_equal(new.count.numpy()[[0, 1, 4]], [0, 0, 0])


if __name__ == "__main__":
    unittest.main()
