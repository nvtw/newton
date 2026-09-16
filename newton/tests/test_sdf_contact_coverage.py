# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Check that mesh edge minima retain the opposing face's support corners."""

import unittest

import numpy as np
import warp as wp

import newton
from newton.tests.unittest_utils import get_test_devices


class TestSDFContactCoverage(unittest.TestCase):
    def test_rotated_box_face_retains_both_sides(self):
        """A vertex's designated edge need not select that vertex as its minimum."""
        for device in get_test_devices():
            mesh = newton.Mesh.create_box(0.05, compute_normals=False, compute_uvs=False)
            wall = newton.Mesh.create_box(0.05, 0.15, 0.15, compute_normals=False, compute_uvs=False)
            if wp.get_device(device).is_cuda:
                mesh.build_sdf(device=device, max_resolution=64)
                wall.build_sdf(device=device, max_resolution=64)
            builder = newton.ModelBuilder(gravity=wp.vec3(0.0))
            rotation = wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), 0.6)
            direction = wp.quat_rotate(rotation, wp.vec3(1.0, 0.0, 0.0))
            for sign, geometry in ((-1.0, mesh), (1.0, wall)):
                body = builder.add_body(xform=wp.transform(sign * 0.0505 * direction, rotation))
                builder.add_shape_mesh(body, mesh=geometry, cfg=newton.ModelBuilder.ShapeConfig(gap=0.001))
            model = builder.finalize(device=device)
            for reduced, deterministic in ((False, False), (True, False), (True, True)):
                with self.subTest(device=device, reduced=reduced, deterministic=deterministic):
                    pipeline = newton.CollisionPipeline(
                        model,
                        reduce_contacts=reduced,
                        deterministic=deterministic,
                        broad_phase="explicit",
                        shape_pairs_filtered=wp.array([[0, 1]], dtype=wp.vec2i, device=device),
                        rigid_contact_max=512,
                    )
                    contacts = pipeline.contacts()
                    state = model.state()
                    pipeline.collide(state, contacts)
                    count = int(contacts.rigid_contact_count.numpy()[0])
                    self.assertGreater(count, 0)
                    self.assertLess(count, 512)
                    points = contacts.rigid_contact_point0.numpy()[:count]
                    self.assertTrue(np.isfinite(points).all())
                    # All four corners of the facing square must be represented.
                    # Merely retaining three points can put the complete support
                    # polygon on one side of the center, creating false rotation.
                    for y in (-0.05, 0.05):
                        for z in (-0.05, 0.05):
                            distance = np.linalg.norm(points[:, 1:] - np.array([y, z]), axis=1)
                            self.assertLess(float(distance.min()), 1e-5, (y, z, points))
                    if reduced:
                        failures = pipeline.narrow_phase.global_contact_reducer.ht_insert_failures.numpy()
                        np.testing.assert_array_equal(failures, 0)


if __name__ == "__main__":
    unittest.main()
