# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Actual CPU matcher controls and explicit negative cases for history retry."""

import unittest

import numpy as np
import warp as wp

from newton._src.geometry.contact_match import ContactMatcher


def fixture(old_x, new_x, new_keys=None, permutation=None):
    """Construct a real sticky matcher over one shape pair without collision generation."""
    n = max(len(old_x), len(new_x), 1)

    def array(x, dtype):
        return wp.array(x, dtype=dtype, device="cpu")

    m = ContactMatcher(
        n, shape_world=array([-1, 0], wp.int32), world_count=1, pos_threshold=0.0005, sticky=True, device="cpu"
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
        "match_index_out": wp.zeros(n, dtype=wp.int32, device="cpu"),
        "device": "cpu",
    }
    if permutation is not None:
        kwargs["canonical_to_source"] = array(permutation, wp.int32)
    return m, kwargs


class TestHistoryMatchInvariants(unittest.TestCase):
    def test_distinct_contacts_survive_permutation(self):
        """Map unique geometry to the same prior identities under current slot permutation."""
        m, k = fixture([0, 0.0004], [0.0004, 0], new_keys=[11, 10], permutation=[1, 0])
        m.match(**k)
        np.testing.assert_array_equal(k["match_index_out"].numpy(), [0, 1])

    def test_nearest_loser_has_valid_unused_alternative(self):
        """Demonstrate the baseline drops a contact despite a compatible unused prior contact."""
        m, k = fixture([0, 0.0004], [0.0001, 0.00015], new_keys=[10, 11])
        m.match(**k)
        matches = k["match_index_out"].numpy()
        self.assertEqual(np.count_nonzero(matches >= 0), 1)
        self.assertLess(abs(0.00015 - 0.0004), np.sqrt(m._pos_threshold_sq))

    def test_duplicate_fingerprints_remain_injective(self):
        """Keep one predecessor owner when current reduced-contact fingerprints collide."""
        m, k = fixture([0, 0.0004], [0.0001, 0.00015], new_keys=[10, 10])
        m.match(**k)
        self.assertEqual(np.count_nonzero(k["match_index_out"].numpy() >= 0), 1)

    def test_disappearance_and_reset_do_not_revive_history(self):
        """Require recontact after an empty prior set or reset to receive new history."""
        for reset in (False, True):
            m, k = fixture([0], [0])
            if reset:
                m.reset()
            else:
                m._prev_count.zero_()
            m.match(**k)
            self.assertLess(k["match_index_out"].numpy()[0], 0)

    def test_incompatible_normal_rebirths(self):
        """Reject a spatially coincident contact whose normal lost correlation."""
        m, k = fixture([0], [0])
        k["normal"].assign(np.array([[1, 0, 0]], dtype=np.float32))
        m.match(**k)
        self.assertLess(k["match_index_out"].numpy()[0], 0)


if __name__ == "__main__":
    unittest.main()
