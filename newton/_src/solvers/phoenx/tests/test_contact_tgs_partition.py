# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Verify lossless grouping, material boundaries and changing membership."""

import unittest

import numpy as np
import warp as wp

from newton._src.solvers.phoenx.constraints.contact_tgs_partition import allocate, partition


def reference(normals, first, count):
    result = []
    for begin, size in zip(first, count, strict=True):
        patches = []
        for point in range(begin, begin + size):
            for members in patches:
                if float(normals[point] @ normals[members[0]]) > 0.999:
                    members.append(point)
                    break
            else:
                patches.append([point])
        result.append(patches)
    return result


def read_groups(state, active):
    first = state.group_first.numpy()
    counts = state.group_count.numpy()
    patch_next = state.patch_next.numpy()
    point_next = state.point_next.numpy()
    point_patch = state.point_patch.numpy()
    patch_first = state.patch_first.numpy()
    patch_count = state.patch_count.numpy()
    result = []
    for group in range(active):
        patches = []
        patch = int(first[group])
        while patch >= 0:
            members = []
            point = int(patch_first[patch])
            while point >= 0:
                assert point_patch[point] == patch
                assert point not in members, "Cyclic point membership"
                members.append(point)
                point = int(point_next[point])
            assert len(members) == patch_count[patch]
            patches.append(members)
            assert len(patches) <= len(point_next), "Cyclic patch membership"
            patch = int(patch_next[patch])
        assert len(patches) == counts[group]
        result.append(patches)
    return result


class TestNormalPatchPartition(unittest.TestCase):
    def test_lossless_material_separation_and_rebuild(self):
        """Partition all contacts without merging incompatible materials."""
        # More than 32 distinct normals verifies that no source-style fixed
        # per-pair buffer silently loses physical contact constraints.
        angle = np.linspace(0, 2 * np.pi, 90, endpoint=False)
        ring = np.stack((np.cos(angle), np.sin(angle), np.zeros_like(angle)), axis=1).astype(np.float32)
        normals = np.concatenate((ring, ring[::-1], ring[:8], ring[:8]))
        first = np.array([0, 180, 188], dtype=np.int32)
        count = np.array([180, 8, 8], dtype=np.int32)
        for device in ["cpu"] + (["cuda:0"] if wp.is_cuda_available() else []):
            with self.subTest(device=device):
                state = allocate(len(normals), 4, device)
                wn = wp.array(normals, dtype=wp.vec3f, device=device)
                wf = wp.array(first, dtype=wp.int32, device=device)
                wc = wp.array(count, dtype=wp.int32, device=device)
                active = wp.array([3], dtype=wp.int32, device=device)
                for sizes in (count, np.array([3, 0, 4], dtype=np.int32), count):
                    wc.assign(sizes)
                    wp.launch(partition, 4, [wn, wf, wc, active, 0.999, state], device=device)
                    actual = read_groups(state, 3)
                    self.assertEqual(actual, reference(normals, first, sizes))
                    flat = [k for group in actual for patch in group for k in patch]
                    self.assertEqual(len(flat), len(set(flat)))
                    self.assertEqual(len(flat), int(sizes.sum()))
                    self.assertEqual(int(state.group_count.numpy()[3]), 0)
                    # Equal normals in separately supplied material groups
                    # remain in separate patches.
                    if sizes[1] and sizes[2]:
                        self.assertNotEqual(state.point_patch.numpy()[180], state.point_patch.numpy()[188])


if __name__ == "__main__":
    unittest.main()
