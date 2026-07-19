# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Regression tests for the scoped current->prev contact warm-start copy."""

import unittest

import numpy as np
import warp as wp

from newton._src.solvers.phoenx.constraints.contact_container import (
    CC_DWORDS_PER_CONTACT,
    CC_IMPULSE_DWORDS_PER_CONTACT,
    CC_PREV_DWORDS_PER_CONTACT,
    contact_container_copy_current_to_prev,
    contact_container_zeros,
)
from newton._src.solvers.phoenx.tests._test_helpers import require_cuda_graph_capture


class TestContactContainerCopy(unittest.TestCase):
    """The copy preserves live warm-start state without touching the inactive tail."""

    def test_copy_is_scoped_to_valid_count(self) -> None:
        device = require_cuda_graph_capture("PhoenX contact-copy tests")
        capacity = 4096
        valid = 37  # Newton packs live contacts into [0, valid).
        cc = contact_container_zeros(capacity, device=device)
        rng = np.random.default_rng(3)
        impulses = rng.standard_normal((CC_IMPULSE_DWORDS_PER_CONTACT, capacity)).astype(np.float32)
        lambdas = rng.standard_normal((CC_DWORDS_PER_CONTACT, capacity)).astype(np.float32)
        cc.impulses.assign(impulses)
        cc.lambdas.assign(lambdas)
        # prev starts non-zero so an out-of-range write would be visible.
        cc.prev_impulses.assign(np.full_like(impulses, -7.0))
        prev_lambdas_initial = np.full((CC_PREV_DWORDS_PER_CONTACT, capacity), -7.0, dtype=np.float32)
        cc.prev_lambdas.assign(prev_lambdas_initial)
        valid_count = wp.array([valid], dtype=wp.int32, device=device)

        with wp.ScopedCapture(device=device) as capture:
            contact_container_copy_current_to_prev(cc, valid_count, device=device)
        wp.capture_launch(capture.graph)

        prev_impulses = cc.prev_impulses.numpy()
        prev_lambdas = cc.prev_lambdas.numpy()
        # Live slots copied exactly.
        np.testing.assert_array_equal(prev_impulses[:, :valid], impulses[:, :valid])
        np.testing.assert_array_equal(prev_lambdas[:, :valid], lambdas[:CC_PREV_DWORDS_PER_CONTACT, :valid])
        # Inactive tail left untouched.
        self.assertTrue(np.all(prev_impulses[:, valid:] == -7.0))
        self.assertTrue(np.all(prev_lambdas[:, valid:] == -7.0))

    def test_zero_valid_count_copies_nothing(self) -> None:
        device = require_cuda_graph_capture("PhoenX contact-copy tests")
        capacity = 512
        cc = contact_container_zeros(capacity, device=device)
        cc.impulses.assign(np.ones((CC_IMPULSE_DWORDS_PER_CONTACT, capacity), dtype=np.float32))
        cc.prev_impulses.assign(np.full((CC_IMPULSE_DWORDS_PER_CONTACT, capacity), -7.0, dtype=np.float32))
        valid_count = wp.zeros(1, dtype=wp.int32, device=device)

        with wp.ScopedCapture(device=device) as capture:
            contact_container_copy_current_to_prev(cc, valid_count, device=device)
        wp.capture_launch(capture.graph)

        self.assertTrue(np.all(cc.prev_impulses.numpy() == -7.0))


if __name__ == "__main__":
    unittest.main()
