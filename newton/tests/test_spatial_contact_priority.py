# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Spatial coverage must distinguish overlap from voxel depth coverage."""

import unittest

import warp as wp

from newton._src.geometry.contact_reduction_global import (
    GlobalContactReducer,
    GlobalContactReducerData,
    _export_and_reduce_contact_centered_two_spatial_depths,
)
from newton.tests.test_contact_reduction_global import get_winning_contacts


@wp.kernel
def register(data: GlobalContactReducerData, x: float, depth: float, fingerprint: int):
    point = wp.vec3(x, 0.0, 0.0)
    _export_and_reduce_contact_centered_two_spatial_depths(
        0,
        1,
        point,
        wp.vec3(0.0, 1.0, 0.0),
        depth,
        fingerprint,
        point,
        0.0,
        0.000212,
        0.0007,
        point,
        wp.vec3(-1.0),
        wp.vec3(1.0),
        wp.vec3i(1),
        data,
        data.deterministic,
    )


@wp.kernel
def register_voxel_coverage_candidates(
    data: GlobalContactReducerData,
    points: wp.array[wp.vec3],
    fingerprints: wp.array[wp.int32],
):
    i = wp.tid()
    point = points[i]
    _export_and_reduce_contact_centered_two_spatial_depths(
        0,
        1,
        point,
        wp.vec3(0.0, 1.0, 0.0),
        -0.01,
        fingerprints[i],
        point,
        0.0,
        0.1,
        0.1,
        point,
        wp.vec3(-1.0),
        wp.vec3(1.0),
        wp.vec3i(5, 1, 5),
        data,
        data.deterministic,
    )


class TestSpatialContactPriority(unittest.TestCase):
    def test_separated_extreme_and_overlap_priority(self):
        """Retain separated extremes while reserving priority for overlap."""
        for device in ["cpu", *(["cuda:0"] if wp.is_cuda_available() else [])]:
            for deterministic in (False, True):
                for reverse in (False, True):
                    with self.subTest(device=device, deterministic=deterministic, reverse=reverse):
                        reducer = GlobalContactReducer(capacity=32, device=device, deterministic=deterministic)
                        data = reducer.get_data_struct()
                        # The nearer separated point supplies depth/voxel coverage.
                        # The farther point supplies a distinct spatial extreme.
                        points = [(0.0, 0.00015, 1), (0.1, 0.0004, 2)]
                        if reverse:
                            points.reverse()
                        for point in points:
                            wp.launch(register, 1, inputs=[data, *point], device=device)

                        def winners(reducer=reducer):
                            ids = get_winning_contacts(reducer)
                            return {int(reducer.contact_fingerprints.numpy()[i]) for i in ids}

                        self.assertEqual(winners(), {1, 2})
                        # An overlapping support point must still outrank both
                        # separated points in all spatial/depth/voxel slots.
                        wp.launch(register, 1, inputs=[data, 0.0, -0.0001, 3], device=device)
                        self.assertEqual(winners(), {3})

    def test_voxel_depth_coverage_can_be_disabled(self):
        """A contact that wins only its voxel slot follows the reducer option."""
        points = [(-0.8, 0.0, 0.0), (0.8, 0.0, 0.0), (0.0, 0.0, -0.8), (0.0, 0.0, 0.8), (0.0, 0.0, 0.0)]
        fingerprints = [10, 11, 12, 13, 1]
        for device in ["cpu", *(["cuda:0"] if wp.is_cuda_available() else [])]:
            results = {}
            for enabled in (True, False):
                reducer = GlobalContactReducer(capacity=32, device=device, voxel_depth_enabled=enabled)
                wp.launch(
                    register_voxel_coverage_candidates,
                    len(points),
                    inputs=[
                        reducer.get_data_struct(),
                        wp.array(points, dtype=wp.vec3, device=device),
                        wp.array(fingerprints, dtype=wp.int32, device=device),
                    ],
                    device=device,
                )
                ids = get_winning_contacts(reducer)
                results[enabled] = {int(reducer.contact_fingerprints.numpy()[i]) for i in ids}
            self.assertIn(1, results[True])
            self.assertNotIn(1, results[False])


if __name__ == "__main__":
    unittest.main()
