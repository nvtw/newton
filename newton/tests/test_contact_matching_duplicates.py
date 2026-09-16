# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Verify injective contact matching when reduced fingerprints collide."""

import unittest

import numpy as np
import warp as wp

from newton._src.geometry.contact_match import ContactMatcher
from newton.tests.unittest_utils import add_function_test, get_test_devices


def fixture(old_x, new_x, new_keys=None, permutation=None, device="cpu"):
    """Construct a real sticky matcher over one shape pair without collision generation."""
    n = max(len(old_x), len(new_x), 1)

    def array(x, dtype):
        return wp.array(x, dtype=dtype, device=device)

    m = ContactMatcher(
        n, shape_world=array([-1, 0], wp.int32), world_count=1, pos_threshold=0.0005, sticky=True, device=device
    )
    keys = np.full(n, np.iinfo(np.int64).max, dtype=np.int64)
    keys[: len(old_x)] = (1 << 23) + np.arange(len(old_x))
    positions = np.zeros((n, 3), dtype=np.float32)
    positions[: len(old_x), 0] = old_x
    normals = np.tile([0.0, 0.0, 1.0], (n, 1)).astype(np.float32)
    m._prev_sorted_keys.assign(keys)
    m._prev_pos_world.assign(positions)
    m._prev_normal.assign(normals)
    m._prev_count.assign(np.array([len(old_x)], dtype=np.int32))
    positions = np.zeros((n, 3), dtype=np.float32)
    positions[: len(new_x), 0] = new_x
    point1 = positions.copy()
    point1[:, 2] = -1e-6
    current = np.arange(n, dtype=np.int64) if new_keys is None else np.asarray(new_keys, dtype=np.int64)
    kwargs = {
        "sort_keys": array((1 << 23) + current, wp.int64),
        "contact_count": array([len(new_x)], wp.int32),
        "point0": array(positions, wp.vec3),
        "point1": array(point1, wp.vec3),
        "shape0": array(np.zeros(n), wp.int32),
        "shape1": array(np.ones(n), wp.int32),
        "normal": array(normals, wp.vec3),
        "margin0": array(np.zeros(n), wp.float32),
        "margin1": array(np.zeros(n), wp.float32),
        "body_q": array([[0, 0, 0, 0, 0, 0, 1]], wp.transform),
        "shape_body": array([-1, -1], wp.int32),
        "match_index_out": wp.zeros(n, dtype=wp.int32, device=device),
        "device": device,
    }
    if permutation is not None:
        kwargs["canonical_to_source"] = array(permutation, wp.int32)
    return m, kwargs


def test_duplicate_contact_keys(test, device):
    """Keep the closest contact when duplicate fingerprints claim one predecessor."""
    m, k = fixture([0, 0.0004], [0.0001, 0.00015], new_keys=[10, 10], device=device)
    m.match(**k)
    matches = k["match_index_out"].numpy()
    test.assertEqual(matches[0], 0)
    test.assertLess(matches[1], 0)


def test_duplicate_contact_key_permutation(test, device):
    """Resolve equal-distance keys by geometry independently of source slot order."""
    winners = []
    for xs in ([0.0001, -0.0001], [-0.0001, 0.0001]):
        m, k = fixture([0, 0.0004], xs, new_keys=[10, 10], device=device)
        m.match(**k)
        ids = np.flatnonzero(k["match_index_out"].numpy() >= 0)
        test.assertEqual(len(ids), 1)
        winners.append(xs[ids[0]])
    test.assertEqual(winners, [-0.0001, -0.0001])


def test_exact_duplicate_contacts(test, device):
    """Retain exactly one owner when contact keys and full geometry are identical."""
    m, k = fixture([0, 0.0004], [0.0001, 0.0001], new_keys=[10, 10], device=device)
    m.match(**k)
    ids = np.flatnonzero(k["match_index_out"].numpy() >= 0)
    np.testing.assert_array_equal(ids, [0])


def test_duplicate_claim_groups_and_active_counts(test, device):
    """Keep predecessor groups independent and discard stale links as counts shrink."""
    m, k = fixture([0, 0.1], [0.0001, 0.00015, 0.1001, 0.10015], new_keys=[10, 10, 10, 10], device=device)
    m.match(**k)
    matches = k["match_index_out"].numpy()
    np.testing.assert_array_equal(matches[[0, 2]], [0, 1])
    test.assertTrue(np.all(matches[[1, 3]] < 0))
    k["contact_count"].assign(np.array([1], dtype=np.int32))
    m.match(**k)
    test.assertEqual(k["match_index_out"].numpy()[0], 0)


class TestDuplicateContactMatching(unittest.TestCase):
    pass


for function in (
    test_duplicate_contact_keys,
    test_duplicate_contact_key_permutation,
    test_exact_duplicate_contacts,
    test_duplicate_claim_groups_and_active_counts,
):
    add_function_test(TestDuplicateContactMatching, function.__name__, function, devices=get_test_devices())


if __name__ == "__main__":
    unittest.main()
