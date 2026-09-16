# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Check CUDA slab averaging against all six physical momentum components."""

import unittest

import numpy as np
import warp as wp

from local_studies.colibri.slab_schedule import build_schedule
from local_studies.colibri.slab_spatial_reference import average_slabs, solve_slabs, solve_slabs_colored


class TestSpatialSlabs(unittest.TestCase):
    def test_internal_contact_rows_and_bounded_motor(self):
        """Unequal copy counts preserve P/L; passive rows cannot add energy."""
        rng = np.random.default_rng(12)
        edges = np.array([[0, 1], [0, 2], [0, 3], [0, 4], [1, 2], [0, 1]], dtype=np.int32)
        x = rng.normal(size=(5, 3)).astype(np.float32)
        v = rng.normal(size=(5, 3)).astype(np.float32)
        w = rng.normal(size=(5, 3)).astype(np.float32)
        masses = np.array([1, 2, 3, 4, 5], dtype=np.float32)
        inertia = np.array([np.diag([0.4, 0.6, 0.8]) * m for m in masses], dtype=np.float32)
        normals = rng.normal(size=(6, 3)).astype(np.float32)
        normals /= np.linalg.norm(normals, axis=1)[:, None]
        points = rng.normal(size=(6, 3)).astype(np.float32)

        def measure(vel, angular):
            p = masses[:, None] * vel.astype(np.float64)
            spin = np.einsum("bij,bj->bi", inertia, angular.astype(np.float64))
            momentum = np.r_[p.sum(axis=0), (spin + np.cross(x, p)).sum(axis=0)]
            energy = 0.5 * (np.sum(p * vel) + np.sum(spin * angular))
            return momentum, energy

        initial_momentum, initial_energy = measure(v, w)
        for motor in (False, True):
            for width in (1, 2, 8):
                with self.subTest(motor=motor, width=width):
                    schedule = build_schedule(edges, 5, width)
                    slabs = schedule.summary()["slabs"]
                    n0, n1 = normals.copy(), -normals.copy()
                    a0 = np.cross(points - x[edges[:, 0]], n0)
                    a1 = np.cross(points - x[edges[:, 1]], n1)
                    lower = np.zeros(6, np.float32)
                    upper = np.full(6, 1e20, np.float32)
                    target = np.zeros(6, np.float32)
                    if motor:
                        n0[-1] = n1[-1] = 0
                        a0[-1] = [0, 0, 1]
                        a1[-1] = [0, 0, -1]
                        lower[-1], upper[-1], target[-1] = -0.001, 0.001, 100
                    membership = np.zeros((slabs, 5), np.int32)
                    for body, active_slabs in enumerate(schedule.body_slabs):
                        membership[active_slabs, body] = 1
                    counts = wp.array(membership.sum(axis=0), dtype=wp.int32, device="cuda:0")
                    copy_v = wp.empty((slabs, 5), dtype=wp.vec3, device="cuda:0")
                    copy_w = wp.empty_like(copy_v)
                    impulse = wp.empty(6, dtype=wp.float32, device="cuda:0")
                    inputs = [
                        wp.array([r for c in schedule.colors for r in c], dtype=wp.int32, device="cuda:0"),
                        wp.array(schedule.row_slab, dtype=wp.int32, device="cuda:0"),
                        wp.array(edges, dtype=wp.vec2i, device="cuda:0"),
                    ]
                    inputs += [wp.array(a, dtype=wp.vec3, device="cuda:0") for a in (n0, a0, n1, a1)]
                    inputs += [
                        wp.array(1 / masses, dtype=wp.float32, device="cuda:0"),
                        wp.array(np.linalg.inv(inertia), dtype=wp.mat33, device="cuda:0"),
                        counts,
                        wp.array(v, dtype=wp.vec3, device="cuda:0"),
                        wp.array(w, dtype=wp.vec3, device="cuda:0"),
                    ]
                    inputs += [wp.array(a, dtype=wp.float32, device="cuda:0") for a in (lower, upper, target)]
                    wp.launch(solve_slabs, slabs, [*inputs, copy_v, copy_w, impulse], device="cuda:0")
                    expected_v, expected_w, expected_impulse = copy_v.numpy(), copy_w.numpy(), impulse.numpy()
                    wp.launch(
                        solve_slabs_colored,
                        (slabs, 32),
                        [
                            wp.array(schedule.row_color, dtype=wp.int32, device="cuda:0"),
                            width,
                            *inputs,
                            copy_v,
                            copy_w,
                            impulse,
                        ],
                        block_dim=32,
                        device="cuda:0",
                    )
                    np.testing.assert_array_equal(copy_v.numpy(), expected_v)
                    np.testing.assert_array_equal(copy_w.numpy(), expected_w)
                    np.testing.assert_array_equal(impulse.numpy(), expected_impulse)
                    result_v = wp.empty(5, dtype=wp.vec3, device="cuda:0")
                    result_w = wp.empty_like(result_v)
                    wp.launch(
                        average_slabs,
                        5,
                        [
                            wp.array(membership, dtype=wp.int32, device="cuda:0"),
                            counts,
                            copy_v,
                            copy_w,
                            result_v,
                            result_w,
                        ],
                        device="cuda:0",
                    )
                    momentum, energy = measure(result_v.numpy(), result_w.numpy())
                    np.testing.assert_allclose(momentum, initial_momentum, rtol=0, atol=3e-6)
                    if motor:
                        self.assertLessEqual(abs(impulse.numpy()[-1]), 0.001000001)
                    else:
                        self.assertLessEqual(energy, initial_energy + 1e-6)


if __name__ == "__main__":
    unittest.main()
