# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Regression tests for the scoped current->prev contact warm-start copy."""

import unittest

import numpy as np
import warp as wp

from newton._src.solvers.phoenx.constraints.contact_container import (
    CC_IMPULSE_DWORDS_PER_CONTACT,
    CC_RIGID_DWORDS_PER_CONTACT,
    contact_container_copy_current_to_prev,
    contact_container_zeros,
)
from newton._src.solvers.phoenx.constraints.contact_ingest import (
    _build_tangent1_from_normal,
    _rotate_tangent_warmstart,
)
from newton._src.solvers.phoenx.tests._test_helpers import require_cuda_graph_capture


@wp.kernel(enable_backward=False)
def _rotate_tangent_warmstart_kernel(
    previous_normal: wp.vec3f,
    previous_tangent1: wp.vec3f,
    previous_lambda1: wp.float32,
    previous_lambda2: wp.float32,
    normal: wp.vec3f,
    tangent1_out: wp.array[wp.vec3f],
    lambda_out: wp.array[wp.vec2f],
):
    tangent1 = _build_tangent1_from_normal(normal)
    tangent1_out[0] = tangent1
    lambda_out[0] = _rotate_tangent_warmstart(
        previous_normal,
        previous_tangent1,
        previous_lambda1,
        previous_lambda2,
        normal,
        tangent1,
    )


class TestContactContainerCopy(unittest.TestCase):
    """The copy preserves live warm-start state without touching the inactive tail."""

    def test_copy_is_scoped_to_valid_count(self) -> None:
        device = require_cuda_graph_capture("PhoenX contact-copy tests")
        capacity = 4096
        valid = 37  # Newton packs live contacts into [0, valid).
        cc = contact_container_zeros(capacity, device=device)
        rng = np.random.default_rng(3)
        impulses = rng.standard_normal((CC_IMPULSE_DWORDS_PER_CONTACT, capacity)).astype(np.float32)
        cc.impulses.assign(impulses)
        # prev starts non-zero so an out-of-range write would be visible.
        cc.prev_impulses.assign(np.full_like(impulses, -7.0))
        lambdas = rng.standard_normal((CC_RIGID_DWORDS_PER_CONTACT, capacity)).astype(np.float32)
        cc.lambdas[:CC_RIGID_DWORDS_PER_CONTACT].assign(lambdas)
        cc.prev_lambdas.assign(np.full_like(lambdas, -9.0))
        valid_count = wp.array([valid], dtype=wp.int32, device=device)

        with wp.ScopedCapture(device=device) as capture:
            contact_container_copy_current_to_prev(cc, valid_count, device=device)
        wp.capture_launch(capture.graph)

        prev_impulses = cc.prev_impulses.numpy()
        prev_lambdas = cc.prev_lambdas.numpy()
        # Live slots copied exactly.
        np.testing.assert_array_equal(prev_impulses[:, :valid], impulses[:, :valid])
        np.testing.assert_array_equal(prev_lambdas[:, :valid], lambdas[:, :valid])
        # Inactive tail left untouched.
        self.assertTrue(np.all(prev_impulses[:, valid:] == -7.0))
        self.assertTrue(np.all(prev_lambdas[:, valid:] == -9.0))

    def test_tangent_warmstart_preserves_world_frame(self) -> None:
        """Preserve a matched friction impulse when its tangent frame rotates."""
        device = require_cuda_graph_capture("PhoenX contact warm-start tests")
        normal_np = np.asarray([0.2, 0.3, np.sqrt(0.87)], dtype=np.float32)
        normal_np /= np.linalg.norm(normal_np)
        tangent1_out = wp.zeros(1, dtype=wp.vec3f, device=device)
        lambda_out = wp.zeros(1, dtype=wp.vec2f, device=device)

        with wp.ScopedCapture(device=device) as capture:
            wp.launch(
                _rotate_tangent_warmstart_kernel,
                dim=1,
                inputs=[
                    wp.vec3f(0.0, 0.0, 1.0),
                    wp.vec3f(1.0, 0.0, 0.0),
                    wp.float32(2.0),
                    wp.float32(3.0),
                    wp.vec3f(*normal_np),
                ],
                outputs=[tangent1_out, lambda_out],
                device=device,
            )
        wp.capture_launch(capture.graph)

        tangent1 = tangent1_out.numpy()[0]
        tangent2 = np.cross(normal_np, tangent1)
        rotated = lambda_out.numpy()[0, 0] * tangent1 + lambda_out.numpy()[0, 1] * tangent2
        previous_world = np.asarray([2.0, 3.0, 0.0], dtype=np.float32)
        expected = previous_world - np.dot(previous_world, normal_np) * normal_np
        np.testing.assert_allclose(rotated, expected, rtol=2.0e-6, atol=2.0e-6)

    def test_zero_valid_count_copies_nothing(self) -> None:
        device = require_cuda_graph_capture("PhoenX contact-copy tests")
        capacity = 512
        cc = contact_container_zeros(capacity, device=device)
        cc.impulses.assign(np.ones((CC_IMPULSE_DWORDS_PER_CONTACT, capacity), dtype=np.float32))
        cc.prev_impulses.assign(np.full((CC_IMPULSE_DWORDS_PER_CONTACT, capacity), -7.0, dtype=np.float32))
        cc.prev_lambdas.assign(np.full((CC_RIGID_DWORDS_PER_CONTACT, capacity), -9.0, dtype=np.float32))
        valid_count = wp.zeros(1, dtype=wp.int32, device=device)

        with wp.ScopedCapture(device=device) as capture:
            contact_container_copy_current_to_prev(cc, valid_count, device=device)
        wp.capture_launch(capture.graph)

        self.assertTrue(np.all(cc.prev_impulses.numpy() == -7.0))
        self.assertTrue(np.all(cc.prev_lambdas.numpy() == -9.0))


if __name__ == "__main__":
    unittest.main()
