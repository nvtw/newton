# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Predictive reduction retains independently closing surface directions."""

import unittest

import numpy as np
import warp as wp

from newton._src.geometry.contact_reduction_global import (
    GlobalContactReducer,
    GlobalContactReducerData,
    export_and_reduce_contact_centered_two_spatial_depths,
    export_and_reduce_predictive_contact,
)
from newton.tests.test_contact_reduction_global import get_winning_contacts


@wp.kernel
def register(
    data: GlobalContactReducerData,
    normals: wp.array[wp.vec3],
    poses: wp.array[wp.transform],
    velocities: wp.array[wp.vec3],
    angular: wp.array[wp.vec3],
    physical: bool,
):
    if physical:
        export_and_reduce_contact_centered_two_spatial_depths(
            0,
            1,
            wp.vec3(0.0),
            normals[1],
            -0.0001,
            200,
            wp.vec3(0.0),
            0.0,
            0.001,
            wp.vec3(0.0),
            wp.vec3(-1.0),
            wp.vec3(1.0),
            wp.vec3i(1),
            data,
        )
    for i in range(129):
        n = normals[0]
        gap = float(0.0002)
        if i == 128:
            n = normals[1]
            gap = 0.0008
        export_and_reduce_predictive_contact(
            0,
            1,
            wp.vec3(float(i) * 0.00001, 0.0, 0.0),
            n,
            gap,
            0.0,
            0.0,
            0.0,
            i,
            poses,
            velocities,
            angular,
            1.0 / 120.0,
            0.005,
            -1,
            data,
        )


class TestPredictiveNormalCoverage(unittest.TestCase):
    def test_closing_flank_survives_nearer_family(self):
        """Keep a closing flank despite nearer candidates occupying all spatial shards."""
        for device in ["cpu", *(["cuda:0"] if wp.is_cuda_available() else [])]:
            for deterministic in (False, True):
                for permutation in ((0, 1, 2), (1, 2, 0), (2, 0, 1)):
                    with self.subTest(device=device, deterministic=deterministic, permutation=permutation):
                        normals = np.array([[0.12, -0.04, 0.99], [-0.89, 0.02, 0.446]], dtype=np.float32)
                        normals /= np.linalg.norm(normals, axis=1)[:, None]
                        normals = normals[:, permutation].copy()
                        velocity = np.array([[0, 0, 0], [0.1, 0, -0.1]], dtype=np.float32)[:, permutation].copy()
                        self.assertTrue(np.all(-normals @ velocity[1] / 120 > np.array([0.0002, 0.0008])))
                        reducer = GlobalContactReducer(
                            capacity=512, device=device, deterministic=deterministic, enable_contact_reclamation=True
                        )
                        wp.launch(
                            register,
                            1,
                            [
                                reducer.get_data_struct(),
                                wp.array(normals, dtype=wp.vec3, device=device),
                                wp.array([wp.transform_identity()] * 2, dtype=wp.transform, device=device),
                                wp.array(velocity, dtype=wp.vec3, device=device),
                                wp.zeros(2, dtype=wp.vec3, device=device),
                                False,
                            ],
                            device=device,
                        )
                        winners = get_winning_contacts(reducer)
                        fingerprints = reducer.contact_fingerprints.numpy()[winners]
                        self.assertIn(128, fingerprints.tolist())
                        wp.launch(
                            register,
                            1,
                            [
                                reducer.get_data_struct(),
                                wp.array(normals, dtype=wp.vec3, device=device),
                                wp.array([wp.transform_identity()] * 2, dtype=wp.transform, device=device),
                                wp.array(velocity, dtype=wp.vec3, device=device),
                                wp.zeros(2, dtype=wp.vec3, device=device),
                                True,
                            ],
                            device=device,
                        )
                        fingerprints = reducer.contact_fingerprints.numpy()[get_winning_contacts(reducer)]
                        self.assertIn(200, fingerprints.tolist())
                        self.assertNotIn(128, fingerprints.tolist())


if __name__ == "__main__":
    unittest.main()
