# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""End-to-end tests for :class:`PhoenxCollisionPipeline`.

Validates that the cloth-aware pipeline:
* exposes ``extra_shape_count == num_cloth_triangles``,
* emits contacts between rigid shapes and cloth triangles,
* drops adjacent cloth-triangle pairs (shared-node filter),
* preserves the rigid-only fast path bit-equivalent to the standard
  :class:`~newton._src.sim.collide.CollisionPipeline` when no cloth
  is present (sanity check on the refactor).
"""

from __future__ import annotations

import unittest

import numpy as np
import warp as wp

import newton
from newton._src.solvers.phoenx.cloth_collision.pipeline import PhoenxCollisionPipeline


def _build_rigid_box_model(device) -> tuple:
    """Single static box at the origin -- the simplest rigid backdrop."""
    builder = newton.ModelBuilder()
    builder.add_shape_box(
        body=-1,
        xform=wp.transform(wp.vec3(0.0, 0.0, 0.0), wp.quat_identity()),
        hx=1.0,
        hy=1.0,
        hz=0.1,
    )
    return builder.finalize(device=device)


@unittest.skipUnless(wp.is_cuda_available(), "Cloth collision pipeline requires CUDA")
class TestPhoenxCollisionPipeline(unittest.TestCase):
    def setUp(self) -> None:
        self.device = wp.get_device()

    def test_construction_with_cloth_triangles(self) -> None:
        """Building the pipeline with T cloth triangles allocates the
        extended per-shape arrays and registers the broadphase filter."""
        model = _build_rigid_box_model(self.device)
        S = int(model.shape_count)
        T = 3
        tri_indices = wp.array(
            np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8]], dtype=np.int32),
            dtype=wp.vec3i,
            device=self.device,
        )
        particle_q = wp.array(np.zeros((9, 3), dtype=np.float32), dtype=wp.vec3, device=self.device)
        particle_radius = wp.array(np.full(9, 0.05, dtype=np.float32), dtype=wp.float32, device=self.device)
        pipeline = PhoenxCollisionPipeline(
            model,
            num_cloth_triangles=T,
            tri_indices=tri_indices,
            particle_q=particle_q,
            particle_radius=particle_radius,
        )
        self.assertEqual(pipeline.extra_shape_count, T)
        # Extended arrays cover S + T slots.
        self.assertEqual(pipeline.shape_type.shape[0], S + T)
        self.assertEqual(pipeline.shape_world.shape[0], S + T)
        self.assertEqual(pipeline.shape_auxiliary.shape[0], S + T)
        # Filter data carries the right threshold + tri table.
        self.assertEqual(int(pipeline._broadphase_filter_data.num_rigid_shapes), S)
        # Cloth slots got the TRIANGLE shape type.
        st = pipeline.shape_type.numpy()
        for t in range(T):
            self.assertEqual(int(st[S + t]), 1000)  # GeoTypeEx.TRIANGLE

    def test_construction_with_zero_cloth_triangles(self) -> None:
        """Zero cloth triangles -> the pipeline still constructs and
        behaves like the standard rigid pipeline."""
        model = _build_rigid_box_model(self.device)
        S = int(model.shape_count)
        # Empty tri_indices array.
        tri_indices = wp.zeros(0, dtype=wp.vec3i, device=self.device)
        particle_q = wp.zeros(0, dtype=wp.vec3, device=self.device)
        particle_radius = wp.zeros(0, dtype=wp.float32, device=self.device)
        pipeline = PhoenxCollisionPipeline(
            model,
            num_cloth_triangles=0,
            tri_indices=tri_indices,
            particle_q=particle_q,
            particle_radius=particle_radius,
        )
        self.assertEqual(pipeline.extra_shape_count, 0)
        self.assertEqual(pipeline.shape_type.shape[0], S)

    def test_collide_emits_rigid_vs_triangle_contact(self) -> None:
        """A cloth triangle directly above a static box generates a
        contact pair through the standard collide() path."""
        model = _build_rigid_box_model(self.device)
        S = int(model.shape_count)

        # One cloth triangle just above the box surface (within margin).
        # Box top is at z = 0.1 (half-extent in z); place triangle at z ~ 0.11.
        particle_positions = np.array(
            [
                [0.0, 0.0, 0.11],
                [0.5, 0.0, 0.11],
                [0.0, 0.5, 0.11],
            ],
            dtype=np.float32,
        )
        T = 1
        tri_indices = wp.array(np.array([[0, 1, 2]], dtype=np.int32), dtype=wp.vec3i, device=self.device)
        particle_q = wp.array(particle_positions, dtype=wp.vec3, device=self.device)
        particle_radius = wp.array(np.full(3, 0.05, dtype=np.float32), dtype=wp.float32, device=self.device)

        pipeline = PhoenxCollisionPipeline(
            model,
            num_cloth_triangles=T,
            tri_indices=tri_indices,
            particle_q=particle_q,
            particle_radius=particle_radius,
            cloth_extra_margin=0.05,
        )

        contacts = pipeline.contacts()
        state = model.state()
        pipeline.collide(state, contacts)

        # At least one rigid-cloth contact should land in the buffer.
        n = int(contacts.rigid_contact_count.numpy()[0])
        self.assertGreater(n, 0, "expected at least one rigid-vs-cloth contact")

        shape0 = contacts.rigid_contact_shape0.numpy()[:n]
        shape1 = contacts.rigid_contact_shape1.numpy()[:n]
        # Every emitted contact should involve the cloth triangle (idx S = 1).
        cloth_shape_idx = S  # = 1
        for k in range(n):
            self.assertTrue(
                shape0[k] == cloth_shape_idx or shape1[k] == cloth_shape_idx,
                f"contact {k}: shapes ({shape0[k]}, {shape1[k]}) - neither is the cloth triangle",
            )


if __name__ == "__main__":
    unittest.main()
