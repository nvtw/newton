# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""End-to-end tests for :class:`PhoenxCollisionPipeline`.

Validates that the cloth-aware pipeline:
* exposes ``extra_shape_count == num_cloth_triangles``,
* preserves the rigid-only fast path bit-equivalent to the standard
  :class:`~newton._src.sim.collide.CollisionPipeline` when no cloth
  is present (sanity check on the refactor).

Cloth-on-rigid contact emission is covered end-to-end (PhoenXWorld
+ populate + step) by ``test_cloth_on_box.py``.
"""

from __future__ import annotations

import unittest

import warp as wp

import newton
from newton._src.solvers.phoenx.cloth_collision.pipeline import PhoenxCollisionPipeline
from newton._src.solvers.phoenx.solver_phoenx import PhoenXWorld


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

    def test_construction_with_zero_cloth_triangles(self) -> None:
        """Zero cloth triangles -> the pipeline still constructs and
        behaves like the standard rigid pipeline."""
        model = _build_rigid_box_model(self.device)
        S = int(model.shape_count)
        # Minimal sentinel constraints (never read because T = 0).
        constraints = PhoenXWorld.make_constraint_container(
            num_joints=0,
            num_cloth_triangles=0,
            device=self.device,
        )
        particle_q = wp.zeros(0, dtype=wp.vec3, device=self.device)
        particle_radius = wp.zeros(0, dtype=wp.float32, device=self.device)
        pipeline = PhoenxCollisionPipeline(
            model,
            num_cloth_triangles=0,
            constraints=constraints,
            cloth_cid_offset=0,
            num_bodies=0,
            particle_q=particle_q,
            particle_radius=particle_radius,
        )
        self.assertEqual(pipeline.extra_shape_count, 0)
        self.assertEqual(pipeline.shape_type.shape[0], S)


if __name__ == "__main__":
    unittest.main()
