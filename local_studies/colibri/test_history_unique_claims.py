# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Fail-before/pass-after controls for duplicate contact fingerprint ownership."""

import unittest

import numpy as np

from .history_unique_claims import UniqueClaims
from .test_history_match_invariants import fixture


class TestUniqueClaims(unittest.TestCase):
    def test_distance_winner_not_every_duplicate_key(self):
        """Keep the closest claimant and demote the further identical-key contact."""
        m, k = fixture([0, 0.0004], [0.0001, 0.00015], new_keys=[10, 10])
        m.match(**k)
        k["match_index_out"].assign(np.array([0, 0], dtype=np.int32))
        UniqueClaims(m).apply(**k)
        self.assertEqual(k["match_index_out"].numpy()[0], 0)
        self.assertLess(k["match_index_out"].numpy()[1], 0)

    def test_equal_distance_geometry_tie_is_permutation_invariant(self):
        """Choose a geometric tie-break consistently when duplicate keys swap source slots."""
        winners = []
        for xs in ([0.0001, -0.0001], [-0.0001, 0.0001]):
            m, k = fixture([0, 0.0004], xs, new_keys=[10, 10])
            m.match(**k)
            UniqueClaims(m).apply(**k)
            ids = np.flatnonzero(k["match_index_out"].numpy() >= 0)
            self.assertEqual(len(ids), 1)
            winners.append(xs[ids[0]])
        self.assertEqual(winners, [-0.0001, -0.0001])

    def test_exact_duplicates_keep_one_owner(self):
        """Retain one predecessor when fresh geometry and keys are exactly identical."""
        m, k = fixture([0, 0.0004], [0.0001, 0.0001], new_keys=[10, 10])
        m.match(**k)
        UniqueClaims(m).apply(**k)
        ids = np.flatnonzero(k["match_index_out"].numpy() >= 0)
        np.testing.assert_array_equal(ids, [0])


if __name__ == "__main__":
    unittest.main()
