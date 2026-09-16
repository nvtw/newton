# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Distinguish geometric search candidates from live physical contacts."""

import unittest

import numpy as np
import warp as wp

import newton
from newton._src.geometry.narrow_phase import NarrowPhase
from newton._src.geometry.types import GeoType


def _check_geometric_narrow_phase(test, device):
    """Check that the simple writer admits receding geometry inside search bounds."""
    shape_transform = wp.array(
        [wp.transform_identity(), wp.transform(wp.vec3(0.3, 0.0, 0.0))],
        dtype=wp.transform,
        device=device,
    )
    shape_aabb_lower = wp.full(2, wp.vec3(-0.1), dtype=wp.vec3, device=device)
    shape_aabb_upper = wp.full(2, wp.vec3(0.1), dtype=wp.vec3, device=device)
    narrow_phase = NarrowPhase(
        max_candidate_pairs=1,
        reduce_contacts=False,
        device=device,
        shape_aabb_lower=shape_aabb_lower,
        shape_aabb_upper=shape_aabb_upper,
        shape_voxel_resolution=wp.full(2, wp.vec3i(1), dtype=wp.vec3i, device=device),
        has_meshes=False,
        contact_max=4,
        verify_buffers=False,
        speculative=True,
        speculative_contact_velocity_filter=False,
    )

    candidate_pair = wp.array([wp.vec2i(0, 1)], dtype=wp.vec2i, device=device)
    candidate_pair_count = wp.array([1], dtype=wp.int32, device=device)
    shape_types = wp.array([int(GeoType.SPHERE), int(GeoType.SPHERE)], dtype=wp.int32, device=device)
    shape_data = wp.array(
        [wp.vec4(0.1, 0.1, 0.1, 0.0), wp.vec4(0.1, 0.1, 0.1, 0.0)],
        dtype=wp.vec4,
        device=device,
    )
    shape_gap = wp.array([0.2, 0.0], dtype=wp.float32, device=device)
    shape_base_gap = wp.zeros(2, dtype=wp.float32, device=device)
    shape_angular_velocity = wp.zeros(2, dtype=wp.vec3, device=device)

    for velocity, expected_count in ((10.0, 1), (-10.0, 1)):
        contact_count = wp.zeros(1, dtype=wp.int32, device=device)
        narrow_phase.launch(
            candidate_pair=candidate_pair,
            candidate_pair_count=candidate_pair_count,
            shape_types=shape_types,
            shape_data=shape_data,
            shape_transform=shape_transform,
            shape_source=wp.zeros(2, dtype=wp.uint64, device=device),
            shape_sdf_index=wp.full(2, -1, dtype=wp.int32, device=device),
            shape_gap=shape_gap,
            shape_base_gap=shape_base_gap,
            shape_collision_radius=wp.full(2, 0.1, dtype=wp.float32, device=device),
            shape_flags=wp.zeros(2, dtype=wp.int32, device=device),
            shape_collision_aabb_lower=shape_aabb_lower,
            shape_collision_aabb_upper=shape_aabb_upper,
            shape_voxel_resolution=wp.full(2, wp.vec3i(1), dtype=wp.vec3i, device=device),
            contact_pair=wp.zeros(4, dtype=wp.vec2i, device=device),
            contact_position=wp.zeros(4, dtype=wp.vec3, device=device),
            contact_normal=wp.zeros(4, dtype=wp.vec3, device=device),
            contact_penetration=wp.zeros(4, dtype=wp.float32, device=device),
            contact_count=contact_count,
            contact_tangent=wp.empty(0, dtype=wp.vec3, device=device),
            shape_linear_velocity=wp.array([wp.vec3(velocity, 0.0, 0.0), wp.vec3(0.0)], dtype=wp.vec3, device=device),
            shape_angular_velocity=shape_angular_velocity,
            collision_update_dt=0.02,
            max_speculative_extension=0.25,
            device=device,
        )
        test.assertEqual(int(contact_count.numpy()[0]), expected_count)


class TestGeometricCandidateAdmission(unittest.TestCase):
    def _check_receding_candidates(self, device):
        """Retain receding search witnesses without changing their true distances."""
        for kind in ("sphere", "box", "convex", "mesh"):
            with self.subTest(kind=kind):
                builder = newton.ModelBuilder(gravity=wp.vec3(0.0))
                cfg = newton.ModelBuilder.ShapeConfig(gap=0.0001)
                for sign in (-1, 1):
                    body = builder.add_body(xform=wp.transform(wp.vec3(sign * 0.0505, 0.0, 0.0), wp.quat_identity()))
                    if kind == "sphere":
                        builder.add_shape_sphere(body, radius=0.05, cfg=cfg)
                    elif kind == "box":
                        builder.add_shape_box(body, hx=0.05, hy=0.05, hz=0.05, cfg=cfg)
                    elif kind == "mesh":
                        builder.add_shape_mesh(body, mesh=newton.Mesh.create_box(0.05), cfg=cfg)
                    else:
                        builder.add_shape_convex_hull(body, mesh=newton.Mesh.create_box(0.05), cfg=cfg)
                model = builder.finalize(device=device)
                state = model.state()
                velocity = state.body_qd.numpy()
                velocity[:, 0] = [-0.2, 0.2]
                state.body_qd.assign(velocity)
                original_gap = model.shape_gap.numpy().copy()
                counts = []
                for filtered, reduced in ((True, True), (False, True), (False, False)):
                    pipeline = newton.CollisionPipeline(
                        model,
                        speculative_contact_gap_max=0.005,
                        speculative_contact_velocity_filter=filtered,
                        reduce_contacts=reduced,
                        rigid_contact_max=64,
                    )
                    contacts = pipeline.contacts()
                    pipeline.collide(state, contacts, dt=1.0 / 120.0)
                    count = int(contacts.rigid_contact_count.numpy()[0])
                    counts.append(count)
                    if count:
                        point0 = contacts.rigid_contact_point0.numpy()[:count] + state.body_q.numpy()[0, :3]
                        point1 = contacts.rigid_contact_point1.numpy()[:count] + state.body_q.numpy()[1, :3]
                        normal = contacts.rigid_contact_normal.numpy()[:count]
                        margins = (
                            contacts.rigid_contact_margin0.numpy()[:count]
                            + contacts.rigid_contact_margin1.numpy()[:count]
                        )
                        np.testing.assert_allclose(
                            np.sum((point1 - point0) * normal, axis=1) - margins, 0.001, atol=2e-6
                        )
                        self.assertTrue(np.isfinite(normal).all())
                        self.assertTrue(np.isfinite(point0).all() and np.isfinite(point1).all())
                self.assertEqual(counts[0], 0)
                self.assertGreater(counts[1], 0)
                self.assertGreater(counts[2], 0)
                np.testing.assert_array_equal(model.shape_gap.numpy(), original_gap)

    def test_receding_candidates_keep_physical_gaps_cpu(self):
        """Check primitive, convex, and mesh writer paths on CPU."""
        self._check_receding_candidates("cpu")
        _check_geometric_narrow_phase(self, "cpu")

    @unittest.skipUnless(wp.is_cuda_available(), "CUDA is unavailable")
    def test_receding_candidates_keep_physical_gaps_cuda(self):
        """Check primitive, convex, and mesh writer paths on CUDA."""
        self._check_receding_candidates("cuda:0")
        _check_geometric_narrow_phase(self, "cuda:0")

    def test_geometric_admission_requires_speculative_search(self):
        """Reject a geometric speculative policy without an enabled search horizon."""
        model = newton.ModelBuilder().finalize(device="cpu")
        with self.assertRaisesRegex(ValueError, "speculative_contact_gap_max"):
            newton.CollisionPipeline(model, speculative_contact_velocity_filter=False)


if __name__ == "__main__":
    unittest.main()
