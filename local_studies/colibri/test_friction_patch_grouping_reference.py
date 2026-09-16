# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Independent graph-rule checks; no collision topology is fabricated."""

import unittest

import numpy as np

from .friction_patch_grouping_reference import group_contacts


def fixture(n):
    return (
        np.tile(np.array([2, 17, 29, 4], np.int32), (n, 1)),
        np.tile(np.array([0, 0, 1], np.float32), (n, 1)),
        np.arange(n, dtype=np.int64),
    )


def identities(groups, ids):
    return [tuple(ids[groups.members[a:b]]) for a, b in zip(groups.offsets[:-1], groups.offsets[1:], strict=True)]


class TestGrouping(unittest.TestCase):
    def test_disconnected_same_key_and_reorder(self):
        keys, normals, ids = fixture(8)
        edges = np.array([[0, 1], [1, 2], [3, 4], [4, 5], [6, 7]], np.int32)
        expected = group_contacts(keys, normals, ids, edges, normal_cosine=0.99)
        self.assertEqual(identities(expected, ids), [(0, 1, 2), (3, 4, 5), (6, 7)])
        rng = np.random.default_rng(5)
        for _ in range(20):
            perm = rng.permutation(8)
            inverse = np.argsort(perm)
            reordered_edges = inverse[edges[rng.permutation(len(edges))]][:, ::-1]
            actual = group_contacts(keys[perm], normals[perm], ids[perm], reordered_edges, normal_cosine=0.99)
            self.assertEqual(identities(actual, ids[perm]), identities(expected, ids))

    def test_normal_chain_does_not_transitively_merge(self):
        keys, normals, ids = fixture(3)
        theta = np.deg2rad(np.array([0, 8, 16], np.float32))
        normals[:, 0] = np.sin(theta)
        normals[:, 2] = np.cos(theta)
        result = group_contacts(
            keys, normals, ids, np.array([[0, 1], [1, 2]], np.int32), normal_cosine=float(np.cos(np.deg2rad(10)))
        )
        self.assertEqual(identities(result, ids), [(0, 1), (2,)])

    def test_world_body_material_and_opposite_normal_separation(self):
        keys, normals, ids = fixture(5)
        keys[1, 0] += 1
        keys[2, 1] += 1
        keys[3, 3] += 1
        normals[4] *= -1
        edges = np.array([(a, b) for a in range(5) for b in range(a + 1, 5)], np.int32)
        result = group_contacts(keys, normals, ids, edges, normal_cosine=0.99)
        self.assertEqual(int(result.counts[0]), 5)

    def test_more_than_128_no_truncation_and_no_edges(self):
        keys, normals, ids = fixture(257)
        edges = np.array([(i, i + 1) for i in range(256)], np.int32)
        result = group_contacts(keys, normals, ids, edges, normal_cosine=0.99)
        np.testing.assert_array_equal(result.counts, [1, 257])
        np.testing.assert_array_equal(result.members, np.arange(257))
        singles = group_contacts(keys, normals, ids, np.empty((0, 2), np.int32), normal_cosine=0.99)
        np.testing.assert_array_equal(singles.counts, [257, 257])

    def test_empty_and_invalid_input(self):
        keys, normals, ids = fixture(0)
        result = group_contacts(keys, normals, ids, np.empty((0, 2), np.int32), normal_cosine=0.99)
        np.testing.assert_array_equal(result.offsets, [0])
        keys, normals, ids = fixture(2)
        with self.assertRaises(ValueError):
            group_contacts(keys, normals, ids, np.array([[0, 2]], np.int32), normal_cosine=0.99)
        with self.assertRaises(ValueError):
            group_contacts(keys, normals, ids * 0, np.empty((0, 2), np.int32), normal_cosine=0.99)
        with self.assertRaises(ValueError):
            group_contacts(keys, normals * 2, ids, np.empty((0, 2), np.int32), normal_cosine=0.99)


if __name__ == "__main__":
    unittest.main()
