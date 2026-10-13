# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Verify cached equality endpoints and ordered body responses."""

import unittest
from types import SimpleNamespace

import numpy as np
import warp as wp

from newton._src.solvers.phoenx.articulations.direct_equality import (
    _apply_direct_equality_descriptor_kernel,
    _build_body_row_wrench_descriptors,
)
from newton._src.solvers.phoenx.body import body_container_zeros


class TestDirectBodyWrenchDescriptors(unittest.TestCase):
    def test_invalid_body_incidence_is_rejected(self):
        """Reject cached row addresses that belong to another body's joint."""
        topology = SimpleNamespace(
            row_joint=np.array([0], dtype=np.int32),
            row_local=np.array([0], dtype=np.int32),
            body_rows=np.array([0], dtype=np.int32),
            body_row_start=np.array([0, 0, 1], dtype=np.int32),
        )
        with self.assertRaisesRegex(ValueError, "joint endpoint"):
            _build_body_row_wrench_descriptors(
                topology,
                np.array([0], dtype=np.int32),
                np.array([-1], dtype=np.int32),
                np.array([1], dtype=np.int32),
                1,
            )

    def test_shuffled_structural_rows_preserve_response_order(self):
        """Preserve endpoint identity, cancellation order and static velocities."""
        parent = np.array([-1, 0, 1, 0], dtype=np.int32)
        child = np.arange(4, dtype=np.int32)
        counts = np.array([3, 1, 6, 2], dtype=np.int32)
        row_joint = np.repeat(np.arange(4, dtype=np.int32), counts)
        row_local = np.concatenate([np.arange(count, dtype=np.int32) for count in counts])
        structural = np.array([2, 0, 3, 1], dtype=np.int32)
        incidence = [[] for _ in range(5)]
        for row, joint in enumerate(row_joint):
            for body in (parent[joint] + 1, child[joint] + 1):
                incidence[body].append(row)
        # Include a noncanonical incidence order independently of row numbering.
        incidence[2].reverse()
        starts = np.concatenate(([0], np.cumsum([len(rows) for rows in incidence]))).astype(np.int32)
        topology = SimpleNamespace(
            row_joint=row_joint,
            row_local=row_local,
            body_rows=np.array([row for rows in incidence for row in rows], dtype=np.int32),
            body_row_start=starts,
        )
        descriptor = _build_body_row_wrench_descriptors(topology, structural, parent, child, 6)
        np.testing.assert_array_equal(descriptor[:, 0], topology.body_rows)

        rng = np.random.default_rng(614)
        wrench0 = rng.integers(-4, 5, size=(4, 6, 6)).astype(np.float32)
        wrench1 = -wrench0.copy()
        scale = np.exp2(np.arange(len(row_joint)) % 3 - 1).astype(np.float32)
        delta = np.exp2(np.arange(len(row_joint)) % 4 - 2).astype(np.float32)
        scale[:3] = 1.0
        delta[:3] = 1.0
        # These terms distinguish ordered accumulation from reassociation.
        wrench1[structural[0], :3, 0] = np.array([2**24, -(2**24), 1], dtype=np.float32)
        expected = np.zeros((5, 6), dtype=np.float32)
        for body, rows in enumerate(incidence):
            for row in rows:
                joint = row_joint[row]
                side = 0 if parent[joint] + 1 == body else 1
                flat = structural[joint] * 6 + row_local[row]
                encoded = descriptor[starts[body] + rows.index(row), 1]
                self.assertEqual(int(encoded), 2 * int(flat) + side)
                value = (wrench0, wrench1)[side][structural[joint], row_local[row]]
                expected[body] = expected[body] + (scale[row] * delta[row]) * value

        with wp.ScopedDevice("cpu"):
            bodies = body_container_zeros(5, device="cpu")
            inverse_mass = np.array([1, 0.5, 0.25, 1, 0], dtype=np.float32)
            bodies.inverse_mass.assign(inverse_mass)
            bodies.inverse_inertia_world.assign(np.tile([0.5, 0.25, 0.125, 0, 0, 0], (5, 1)))
            initial = np.full((5, 3), 0.125, dtype=np.float32)
            bodies.velocity.assign(initial)
            bodies.angular_velocity.assign(initial)
            wp.launch(
                _apply_direct_equality_descriptor_kernel,
                dim=5,
                inputs=[
                    wp.array(starts, dtype=wp.int32, device="cpu"),
                    wp.array(descriptor, dtype=wp.vec2i, device="cpu"),
                    wp.array(wrench0.reshape(-1, 6), dtype=wp.spatial_vector, device="cpu"),
                    wp.array(wrench1.reshape(-1, 6), dtype=wp.spatial_vector, device="cpu"),
                    wp.array(delta, device="cpu"),
                    wp.array(scale, device="cpu"),
                    bodies,
                ],
                device="cpu",
            )
            linear = initial + inverse_mass[:, None] * expected[:, :3]
            angular = initial + np.array([0.5, 0.25, 0.125], dtype=np.float32) * expected[:, 3:]
            linear[[0, 4]] = initial[[0, 4]]
            angular[[0, 4]] = initial[[0, 4]]
            np.testing.assert_array_equal(bodies.velocity.numpy(), linear)
            np.testing.assert_array_equal(bodies.angular_velocity.numpy(), angular)


if __name__ == "__main__":
    unittest.main()
