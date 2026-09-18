# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Verify the self-contained Caterpillar descriptor and extracted meshes."""

import unittest
from collections import Counter

import numpy as np
import warp as wp

from newton.examples.phoenx.caterpillar_scene import SCENE
from newton.examples.phoenx.example_phoenx_caterpillar import ASSETS, Example, _load_mesh, _transform


class TestCaterpillar(unittest.TestCase):
    def test_real_scale_and_topology(self):
        """Retain the full-scale excavator and both closed 46-link tracks."""
        self.assertEqual(SCENE["meters_per_unit"], 1.0)
        self.assertEqual(len(SCENE["bodies"]), 131)
        self.assertEqual(
            Counter(joint["type"] for joint in SCENE["joints"]),
            {"revolute": 129, "prismatic": 3, "fixed": 2},
        )
        origins = np.asarray([body["pose"][:3] for body in SCENE["bodies"]])
        np.testing.assert_allclose(np.ptp(origins, axis=0), [6.97355, 6.54615, 3.11695], rtol=0.0, atol=1.0e-4)
        for side in ("left", "right"):
            links = {index for index, body in enumerate(SCENE["bodies"]) if f"/{side}_track_chain/" in body["label"]}
            self.assertEqual(len(links), 46)
            degree = dict.fromkeys(links, 0)
            for joint in SCENE["joints"]:
                if joint["parent"] in links and joint["child"] in links:
                    degree[joint["parent"]] += 1
                    degree[joint["child"]] += 1
            self.assertEqual(set(degree.values()), {2})

    def test_joint_frames(self):
        """Resolve descendant joint targets without changing their world frames."""
        bodies = SCENE["bodies"]
        for joint in SCENE["joints"]:
            frames = []
            for body, values in zip((joint["parent"], joint["child"]), joint["frames"], strict=True):
                frame = _transform(values)
                if body >= 0:
                    frame = _transform(bodies[body]["pose"]) * frame
                frames.append(frame)
            parent, child = frames
            self.assertLess(
                float(wp.length(wp.transform_get_translation(parent) - wp.transform_get_translation(child))),
                2.0e-6,
                joint["label"],
            )
            q = wp.mul(wp.quat_inverse(wp.transform_get_rotation(parent)), wp.transform_get_rotation(child))
            self.assertLess(float(wp.length(wp.vec3(q[0], q[1], q[2]))), 5.0e-6, joint["label"])

    def test_collision_and_drive_data(self):
        """Preserve SDF collisions, hydraulic joints, and converted angular targets."""
        collisions = [shape for shape in SCENE["shapes"] if shape["collision"]]
        self.assertEqual(len(collisions), 315)
        self.assertEqual(
            Counter(shape["approximation"] for shape in collisions),
            {"sdf": 314, "convexHull": 1},
        )
        self.assertTrue(all(shape["density"] == 1000.0 for shape in collisions))
        hydraulics = [joint for joint in SCENE["joints"] if joint["type"] == "prismatic"]
        self.assertEqual(len(hydraulics), 3)
        driven = [joint for joint in SCENE["joints"] if joint["drive_enabled"]]
        self.assertEqual(len(driven), 6)
        arm = next(joint for joint in driven if joint["label"].endswith("/ArmBaseJoint"))
        self.assertAlmostEqual(arm["target"], np.deg2rad(10.0))
        self.assertAlmostEqual(arm["stiffness"], 100000.0 * 180.0 / np.pi)

    def test_mesh_quality(self):
        """Retain finite indexed triangles and normalized explicit normals."""
        first = ASSETS / SCENE["shapes"][0]["mesh"]
        if not first.exists():
            self.skipTest("copyrighted Caterpillar OBJ meshes are local-only")
        filenames = sorted({shape["mesh"] for shape in SCENE["shapes"]})
        self.assertEqual(len(filenames), 453)
        for filename in filenames:
            with self.subTest(mesh=filename), wp.ScopedDevice("cpu"):
                mesh = _load_mesh(ASSETS / filename, 0.5)
                self.assertTrue(np.isfinite(mesh.vertices).all())
                self.assertEqual(len(mesh.indices) % 3, 0)
                self.assertGreaterEqual(int(mesh.indices.min()), 0)
                self.assertLess(int(mesh.indices.max()), len(mesh.vertices))
                self.assertEqual(mesh.normals.shape, mesh.vertices.shape)
                np.testing.assert_allclose(np.linalg.norm(mesh.normals, axis=1), 1.0, atol=2.0e-5)

    def test_measured_solver_defaults(self):
        """Keep the measured 40 FPS configuration."""
        args = Example.create_parser().parse_args([])
        self.assertEqual(args.substeps, 5)
        self.assertEqual(args.iterations, 2)


if __name__ == "__main__":
    unittest.main()
