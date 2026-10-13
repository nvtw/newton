# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Regression checks for the split narrow-phase box GJK queue filter."""

import math
import unittest
from unittest.mock import patch

import numpy as np
import warp as wp

import newton
from newton._src.geometry.narrow_phase import NarrowPhase
from newton._src.geometry.types import GeoType
from newton._src.sim import collide as collide_module


@unittest.skipUnless(wp.is_cuda_available(), "Split GJK/MPR requires CUDA")
class TestBoxGjkQueueFilter(unittest.TestCase):
    def setUp(self):
        self.device = wp.get_cuda_devices()[0]
        rotated = wp.quat(0.0, 0.0, math.sin(math.pi / 8.0), math.cos(math.pi / 8.0))
        identity = wp.quat_identity()
        self.shape_types = wp.array([int(GeoType.BOX)] * 4 + [int(GeoType.SPHERE)], dtype=int, device=self.device)
        self.shape_data = wp.array(
            [
                wp.vec4(1.0, 1.0, 1.0, 0.0),
                wp.vec4(1.0, 1.0, 1.0, 0.0),
                wp.vec4(1.0, 1.0, 1.0, 0.0),
                wp.vec4(1.0, 1.0, 1.0, 0.0),
                wp.vec4(0.5, 0.0, 0.0, 0.0),
            ],
            dtype=wp.vec4,
            device=self.device,
        )
        self.shape_transform = wp.array(
            [
                wp.transform(wp.vec3(0.0, 0.0, 0.0), identity),
                wp.transform(wp.vec3(2.35, 0.0, 0.0), rotated),
                wp.transform(wp.vec3(6.0, 0.0, 0.0), identity),
                wp.transform(wp.vec3(2.47, 0.0, 0.0), rotated),
                wp.transform(wp.vec3(1.54, 0.0, 0.0), identity),
            ],
            dtype=wp.transform,
            device=self.device,
        )
        self.candidate_pairs = wp.array([(0, 1), (0, 2), (0, 3), (0, 4)], dtype=wp.vec2i, device=self.device)
        self.candidate_count = wp.array([4], dtype=int, device=self.device)
        self.shape_gap = wp.array([0.05] * 5, dtype=float, device=self.device)
        self.shape_base_gap = wp.array([0.01] * 5, dtype=float, device=self.device)
        self.shape_source = wp.zeros(5, dtype=wp.uint64, device=self.device)
        self.shape_radius = wp.array([math.sqrt(3.0)] * 4 + [0.5], dtype=float, device=self.device)
        self.shape_flags = wp.zeros(5, dtype=int, device=self.device)
        self.shape_sdf_index = wp.array([-1] * 5, dtype=int, device=self.device)
        self.collision_aabb_lower = wp.zeros(5, dtype=wp.vec3, device=self.device)
        self.collision_aabb_upper = wp.zeros(5, dtype=wp.vec3, device=self.device)
        self.shape_voxel_resolution = wp.zeros(5, dtype=wp.vec3i, device=self.device)
        self.linear_velocities = wp.array(
            [wp.vec3(0.0, 0.0, 0.0)] * 3 + [wp.vec3(-4.0, 0.0, 0.0)] * 2,
            dtype=wp.vec3,
            device=self.device,
        )
        self.angular_velocities = wp.zeros(5, dtype=wp.vec3, device=self.device)

    def make_narrow_phase(self, *, filter_box_pairs):
        return NarrowPhase(
            max_candidate_pairs=8,
            candidate_pair_work_estimate=8,
            max_triangle_pairs=0,
            device=self.device,
            shape_aabb_lower=self.collision_aabb_lower,
            shape_aabb_upper=self.collision_aabb_upper,
            has_meshes=False,
            has_heightfields=False,
            use_lean_gjk_mpr=True,
            convex_support_acceleration=False,
            split_gjk_mpr=True,
            _filter_split_gjk_box_pairs=filter_box_pairs,
            speculative=True,
            deterministic=True,
            contact_max=32,
        )

    def make_outputs(self):
        return {
            "contact_pair": wp.empty(32, dtype=wp.vec2i, device=self.device),
            "contact_position": wp.empty(32, dtype=wp.vec3, device=self.device),
            "contact_normal": wp.empty(32, dtype=wp.vec3, device=self.device),
            "contact_penetration": wp.empty(32, dtype=float, device=self.device),
            "contact_tangent": wp.empty(32, dtype=wp.vec3, device=self.device),
            "contact_count": wp.zeros(1, dtype=int, device=self.device),
        }

    def launch_into(self, narrow_phase, outputs):
        narrow_phase.launch(
            candidate_pair=self.candidate_pairs,
            candidate_pair_count=self.candidate_count,
            shape_types=self.shape_types,
            shape_data=self.shape_data,
            shape_transform=self.shape_transform,
            shape_source=self.shape_source,
            shape_gap=self.shape_gap,
            shape_base_gap=self.shape_base_gap,
            shape_collision_radius=self.shape_radius,
            shape_flags=self.shape_flags,
            shape_sdf_index=self.shape_sdf_index,
            shape_collision_aabb_lower=self.collision_aabb_lower,
            shape_collision_aabb_upper=self.collision_aabb_upper,
            shape_voxel_resolution=self.shape_voxel_resolution,
            shape_linear_velocity=self.linear_velocities,
            shape_angular_velocity=self.angular_velocities,
            collision_update_dt=1.0 / 60.0,
            max_speculative_extension=1.0,
            device=self.device,
            **outputs,
        )

    @staticmethod
    def read_contacts(outputs):
        count = int(outputs["contact_count"].numpy()[0])
        return {name: array[:count].numpy() for name, array in outputs.items() if name != "contact_count"}

    def test_contacts_match_and_gjk_work_decreases(self):
        """Preserve contact bytes while removing separated GJK work."""
        baseline = self.make_narrow_phase(filter_box_pairs=False)
        filtered = self.make_narrow_phase(filter_box_pairs=True)
        baseline_outputs = self.make_outputs()
        filtered_outputs = self.make_outputs()
        self.launch_into(baseline, baseline_outputs)
        self.launch_into(filtered, filtered_outputs)
        baseline_contacts = self.read_contacts(baseline_outputs)
        filtered_contacts = self.read_contacts(filtered_outputs)
        self.assertGreater(len(baseline_contacts["contact_pair"]), 0)
        for expected_pair in ((0, 3), (0, 4)):
            pairs = baseline_contacts["contact_pair"]
            rows = np.all(pairs == expected_pair, axis=1) | np.all(pairs == expected_pair[::-1], axis=1)
            self.assertTrue(np.any(rows), f"Missing predicted contact for {expected_pair}")
            self.assertTrue(np.any(baseline_contacts["contact_penetration"][rows] > 0.0))
        for name, expected in baseline_contacts.items():
            np.testing.assert_array_equal(filtered_contacts[name], expected, err_msg=name)
        original_work = int(filtered.split_gjk_work_count.numpy()[0])
        filtered_work = int(filtered.filtered_gjk_work_count.numpy()[0])
        self.assertGreater(original_work, 0)
        self.assertLess(filtered_work, original_work)
        self.assertEqual(original_work, int(baseline.split_gjk_work_count.numpy()[0]))

        # Primitive routing handles sphere-box directly, so feed a mixed pair
        # to the filter explicitly to verify its cheap non-box pass-through.
        original_items = wp.array([0, 1, 2, 3], dtype=int, device=self.device)
        original_count = wp.array([4], dtype=int, device=self.device)
        filtered_items = wp.empty(4, dtype=int, device=self.device)
        filtered_count = wp.zeros(1, dtype=int, device=self.device)
        wp.launch(
            filtered.filter_split_gjk_box_pairs_kernel,
            dim=32,
            inputs=[
                original_items,
                original_count,
                self.candidate_pairs,
                self.shape_types,
                self.shape_data,
                self.shape_transform,
                self.shape_source,
                self.shape_gap,
                self.shape_radius,
                self.collision_aabb_lower,
                self.collision_aabb_upper,
                self.collision_aabb_lower,
                self.collision_aabb_upper,
                32,
                filtered_items,
                filtered_count,
            ],
            block_dim=32,
            device=self.device,
        )
        self.assertEqual(int(filtered_count.numpy()[0]), 3)
        np.testing.assert_array_equal(np.sort(filtered_items[:3].numpy()), [0, 2, 3])

    def test_graph_replay_resets_filter_and_contacts(self):
        """Reset consolidated counters when replaying the complete narrow phase."""
        narrow_phase = self.make_narrow_phase(filter_box_pairs=True)
        outputs = self.make_outputs()
        self.launch_into(narrow_phase, outputs)
        expected_count = int(narrow_phase.filtered_gjk_work_count.numpy()[0])
        expected_contacts = self.read_contacts(outputs)
        with wp.ScopedCapture() as capture:
            outputs["contact_count"].zero_()
            self.launch_into(narrow_phase, outputs)
        for _ in range(2):
            wp.capture_launch(capture.graph)
            self.assertEqual(int(narrow_phase.filtered_gjk_work_count.numpy()[0]), expected_count)
            actual_contacts = self.read_contacts(outputs)
            for name, expected in expected_contacts.items():
                np.testing.assert_array_equal(actual_contacts[name], expected, err_msg=name)

    def test_pipeline_dispatch_requires_box_box_candidates(self):
        """Skip automatic filtering for split scenes without box-box candidates."""

        def make_model(shape_types):
            builder = newton.ModelBuilder()
            for index, shape_type in enumerate(shape_types):
                body = builder.add_body(xform=wp.transform(wp.vec3(float(index) * 2.0, 0.0, 0.0)))
                if shape_type == GeoType.BOX:
                    builder.add_shape_box(body, hx=0.5, hy=0.5, hz=0.5)
                else:
                    builder.add_shape_cone(body, radius=0.5, half_height=0.5)
            return builder.finalize(device=self.device)

        with (
            patch.object(collide_module, "_SPLIT_GJK_MPR_LEAN_PAIR_COUNT_THRESHOLD", 1),
            patch.object(collide_module, "_SPLIT_GJK_MPR_FULL_PAIR_COUNT_THRESHOLD", 1),
        ):
            for shape_types, expected_filter in (
                ((GeoType.BOX, GeoType.BOX), True),
                ((GeoType.BOX, GeoType.CONE), False),
                ((GeoType.CONE, GeoType.CONE), False),
            ):
                with self.subTest(shape_types=shape_types):
                    pipeline = newton.CollisionPipeline(make_model(shape_types), broad_phase="nxn")
                    self.assertTrue(pipeline.narrow_phase.split_gjk_mpr)
                    self.assertEqual(pipeline.narrow_phase.box_face_sat_gate, expected_filter)
                    self.assertEqual(pipeline.narrow_phase.filtered_gjk_work_items is not None, expected_filter)

    def test_explicit_virtual_pair_does_not_index_model_shape_types(self):
        """An explicit virtual suffix pair remains valid during split-path setup."""
        builder = newton.ModelBuilder()
        for index in range(2):
            body = builder.add_body(xform=wp.transform(wp.vec3(float(index) * 2.0, 0.0, 0.0)))
            builder.add_shape_box(body, hx=0.5, hy=0.5, hz=0.5)
        model = builder.finalize(device=self.device)
        shape_world = model.shape_world.numpy()
        shape_flags = model.shape_flags.numpy()
        unified_world = wp.array(np.append(shape_world, shape_world[0]), dtype=int, device=self.device)
        unified_flags = wp.array(np.append(shape_flags, shape_flags[0]), dtype=int, device=self.device)
        explicit_pairs = wp.array([(0, 2)], dtype=wp.vec2i, device=self.device)
        with patch.object(collide_module, "_SPLIT_GJK_MPR_LEAN_PAIR_COUNT_THRESHOLD", 1):
            pipeline = newton.CollisionPipeline(
                model,
                broad_phase="explicit",
                shape_pairs_filtered=explicit_pairs,
                extra_shape_count=1,
                unified_shape_world=unified_world,
                unified_shape_flags=unified_flags,
            )
        self.assertTrue(pipeline.narrow_phase.split_gjk_mpr)
        self.assertTrue(pipeline.narrow_phase.box_face_sat_gate)


if __name__ == "__main__":
    unittest.main()
