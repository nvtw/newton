# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Verify retry eligibility and history ownership through actual CPU kernels."""

import unittest

import numpy as np
import warp as wp

from .history_match_retry import Retry
from .test_history_match_invariants import fixture


class TestHistoryRetry(unittest.TestCase):
    def test_recover_unused_candidate(self):
        """Recover an eligible loser without moving the original winner."""
        m, k = fixture([0, 0.0004], [0.0001, 0.00015], new_keys=[10, 11])
        m.match(**k)
        self.assertLess(k["match_index_out"].numpy()[1], 0)
        retry = Retry(m)
        retry.apply(**k)
        np.testing.assert_array_equal(k["match_index_out"].numpy(), [0, 1])
        self.assertEqual(retry.stats.numpy()[4], 0)

    def test_permuted_source_preserves_physical_ownership(self):
        """Keep predecessor ownership stable under a source-slot permutation."""
        for xs, keys, perm in [([0.0001, 0.00015], [10, 11], [0, 1]), ([0.00015, 0.0001], [11, 10], [1, 0])]:
            m, k = fixture([0, 0.0004], xs, new_keys=keys, permutation=perm)
            m.match(**k)
            Retry(m).apply(**k)
            np.testing.assert_array_equal(k["match_index_out"].numpy(), [0, 1])

    def test_reset_disappearance_and_normal_rejection(self):
        """Prevent retry from resurrecting invalidated or geometrically incompatible history."""
        for mode in ("reset", "disappear", "normal", "world_reset", "gap"):
            m, k = fixture([0], [0])
            if mode == "reset":
                m.reset()
            elif mode == "disappear":
                m._prev_count.zero_()
            elif mode == "normal":
                k["normal"].assign(np.array([[1, 0, 0]], dtype=np.float32))
            elif mode == "world_reset":
                m.reset(wp.array([True, False], dtype=wp.bool, device="cpu"))
            else:
                k["point1"].assign(np.array([[0, 0, 0.003]], dtype=np.float32))
            m.match(**k)
            Retry(m, dormant=True).apply(**k)
            self.assertLess(k["match_index_out"].numpy()[0], 0, mode)

    def test_duplicate_geometry_is_injective_after_retry(self):
        """Give duplicate geometry separate predecessors without copying one history twice."""
        m, k = fixture([0, 0.0004], [0.0001, 0.0001], new_keys=[10, 10])
        m.match(**k)
        Retry(m).apply(**k)
        np.testing.assert_array_equal(k["match_index_out"].numpy(), [0, 1])


if __name__ == "__main__":
    unittest.main()
