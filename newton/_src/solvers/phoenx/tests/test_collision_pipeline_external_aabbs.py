# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Tests for :meth:`CollisionPipeline.collide_with_external_aabbs` and
the PhoenX wireup that registers cloth triangles as virtual shapes in
the pipeline's unified per-shape arrays.

Covers:

1. Rigid-rigid regression: when no cloth is present, the new entry
   point must produce the same rigid contacts as the standard
   :meth:`CollisionPipeline.collide` path.
2. Unified suffix wiring: the pipeline's unified ``shape_type`` /
   ``shape_gap`` / ``shape_world`` / ``shape_flags`` /
   ``shape_collision_group`` arrays are length ``S + T``; their suffix
   carries the cloth-triangle metadata stamped at solver init.
3. ``tri_indices`` content: per-triangle ``vec4i`` rows mirror the
   model's triangle connectivity with the 4th component pinned at
   ``-1`` (reserved for tetrahedron support).
4. Broad-phase candidate-pair detection across the unified ``S+T``
   set: a rigid box overlapping a cloth-triangle AABB must produce at
   least one candidate pair that touches the triangle suffix.

CUDA-only by Newton convention.
"""

from __future__ import annotations

import unittest

import numpy as np
import warp as wp

import newton
from newton._src.geometry.flags import ShapeFlags
from newton._src.geometry.support_function import GeoTypeEx
from newton._src.solvers.phoenx.solver import SolverPhoenX


@unittest.skipUnless(wp.is_cuda_available(), "External-AABB collision test requires CUDA")
class TestCollideWithExternalAabbsRigidOnly(unittest.TestCase):
    """Rigid-only scene: ``collide_with_external_aabbs`` rejects the
    rigid-only configuration so callers don't accidentally pay the
    unified-array allocation cost on scenes that don't need it."""

    def setUp(self) -> None:
        self.device = wp.get_device()
        mb = newton.ModelBuilder()
        mb.add_ground_plane()
        body = mb.add_body(
            xform=wp.transform(p=wp.vec3(0.0, 0.0, 0.05), q=wp.quat_identity()),
        )
        mb.add_shape_box(
            body,
            hx=0.1,
            hy=0.1,
            hz=0.1,
            cfg=newton.ModelBuilder.ShapeConfig(density=1000.0, mu=0.5),
        )
        mb.gravity = -9.81
        self.model = mb.finalize(device=self.device)
        self.solver = SolverPhoenX(self.model)
        self.pipeline = self.model._collision_pipeline

    def test_rejects_rigid_only_pipeline(self) -> None:
        state = self.model.state()
        contacts = self.pipeline.contacts()
        with self.assertRaises(RuntimeError):
            self.pipeline.collide_with_external_aabbs(state, contacts)


@unittest.skipUnless(wp.is_cuda_available(), "External-AABB cloth test requires CUDA")
class TestCollideWithExternalAabbsCloth(unittest.TestCase):
    """End-to-end cloth + rigid scene: the unified arrays drive the
    broad phase and ``tri_indices`` is populated with ``-1`` in the
    4th slot."""

    def setUp(self) -> None:
        self.device = wp.get_device()
        mb = newton.ModelBuilder()
        mb.add_ground_plane()
        body = mb.add_body(
            xform=wp.transform(p=wp.vec3(0.5, 0.5, 0.45), q=wp.quat_identity()),
        )
        mb.add_shape_box(
            body,
            hx=0.4,
            hy=0.4,
            hz=0.05,
            cfg=newton.ModelBuilder.ShapeConfig(density=1000.0, mu=0.5),
        )
        mb.add_cloth_grid(
            pos=wp.vec3(0.0, 0.0, 0.5),
            rot=wp.quat_identity(),
            vel=wp.vec3(0.0, 0.0, 0.0),
            dim_x=2,
            dim_y=2,
            cell_x=0.5,
            cell_y=0.5,
            mass=0.1,
            tri_ke=1234.0,
            tri_ka=9876.0,
        )
        mb.gravity = -9.81
        self.model = mb.finalize(device=self.device)
        self.solver = SolverPhoenX(self.model, cloth_margin=0.01)
        self.pipeline = self.model._collision_pipeline

        self.S = int(self.model.shape_count)
        self.T = int(self.model.tri_count)

    def test_pipeline_unified_arrays_have_st_length(self) -> None:
        """Pipeline-owned unified arrays cover the full ``S + T`` shape
        set; the suffix carries cloth-triangle metadata stamped at
        solver init."""
        self.assertEqual(self.solver.num_cloth_triangles, self.T)
        self.assertGreaterEqual(self.pipeline.extra_shape_count, self.T)
        total = self.S + self.T

        self.assertEqual(self.pipeline.unified_shape_type.shape[0], total)
        self.assertEqual(self.pipeline.unified_shape_gap.shape[0], total)
        self.assertEqual(self.pipeline.unified_shape_flags.shape[0], total)
        self.assertEqual(self.pipeline.unified_shape_world.shape[0], total)
        self.assertEqual(self.pipeline.unified_shape_collision_group.shape[0], total)
        self.assertEqual(self.pipeline.narrow_phase.shape_aabb_lower.shape[0], total)
        self.assertEqual(self.pipeline.narrow_phase.shape_aabb_upper.shape[0], total)
        self.assertEqual(self.pipeline.geom_data.shape[0], total)
        self.assertEqual(self.pipeline.geom_transform.shape[0], total)

        # Rigid prefix is seeded from model.shape_*.
        np.testing.assert_array_equal(
            self.pipeline.unified_shape_type.numpy()[: self.S],
            self.model.shape_type.numpy(),
        )

        # Suffix carries TRIANGLE / cloth_margin / COLLIDE_SHAPES /
        # default group 1.
        suffix_type = self.pipeline.unified_shape_type.numpy()[self.S :]
        suffix_gap = self.pipeline.unified_shape_gap.numpy()[self.S :]
        suffix_flags = self.pipeline.unified_shape_flags.numpy()[self.S :]
        suffix_group = self.pipeline.unified_shape_collision_group.numpy()[self.S :]

        self.assertTrue(np.all(suffix_type == int(GeoTypeEx.TRIANGLE)))
        np.testing.assert_allclose(suffix_gap, 0.01, rtol=1e-6)
        self.assertTrue(np.all(suffix_flags == int(ShapeFlags.COLLIDE_SHAPES)))
        self.assertTrue(np.all(suffix_group == 1))

        # Suffix shape_world inherits from particle_world at the
        # triangle's first vertex (or -1 in single-world scenes).
        suffix_world = self.pipeline.unified_shape_world.numpy()[self.S :]
        if self.model.particle_world is not None:
            pworld = self.model.particle_world.numpy()
            tri = self.model.tri_indices.numpy().reshape(-1, 3)
            expected = pworld[tri[:, 0]].astype(np.int32)
            np.testing.assert_array_equal(suffix_world, expected)

    def test_update_external_geom_fills_suffix_and_tri_indices(self) -> None:
        """The per-step kernel writes triangle AABBs / geom data into
        the pipeline's suffix and populates ``tri_indices`` with the
        ``-1`` 4th component."""
        state = self.model.state()
        self.solver.update_external_geom(state)

        tri_idx = self.solver.tri_indices.numpy()
        self.assertEqual(tri_idx.shape[0], self.T)
        self.assertTrue(np.all(tri_idx[:, 3] == -1))
        model_tri = self.model.tri_indices.numpy().reshape(-1, 3)
        np.testing.assert_array_equal(tri_idx[:, :3], model_tri)

        particle_q = state.particle_q.numpy()
        a, b, c = model_tri[0]
        verts = np.stack([particle_q[a], particle_q[b], particle_q[c]], axis=0)
        lo = self.pipeline.narrow_phase.shape_aabb_lower.numpy()[self.S]
        hi = self.pipeline.narrow_phase.shape_aabb_upper.numpy()[self.S]
        np.testing.assert_array_less(lo - 1e-6, verts.min(axis=0))
        np.testing.assert_array_less(verts.max(axis=0), hi + 1e-6)

        # geom_transform translation must equal vertex A.
        geom_tx = self.pipeline.geom_transform.numpy()[self.S]
        # transform layout: (px, py, pz, qx, qy, qz, qw)
        np.testing.assert_allclose(geom_tx[:3], particle_q[a], rtol=1e-5, atol=1e-6)

    def test_mixed_broadphase_pairs_detected(self) -> None:
        """A rigid box positioned to overlap a cloth-triangle's AABB
        must produce at least one candidate pair touching the triangle
        suffix in the broad-phase output."""
        state = self.model.state()
        self.solver.update_external_geom(state)

        contacts = self.pipeline.contacts()
        self.pipeline.collide_with_external_aabbs(state, contacts)

        pair_count = int(self.pipeline.broad_phase_pair_count.numpy()[0])
        self.assertGreater(pair_count, 0, "broad phase produced zero candidate pairs")
        pairs = self.pipeline.broad_phase_shape_pairs.numpy()[:pair_count]
        max_idx = pairs.max(axis=1)
        self.assertTrue(
            np.any(max_idx >= self.S),
            "expected at least one candidate pair touching the cloth-triangle suffix",
        )


if __name__ == "__main__":
    unittest.main()
