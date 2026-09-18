# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Verify the self-contained analog-digital clock asset and authored joint frames."""

import unittest
from unittest.mock import patch

import numpy as np
import warp as wp

import newton
from newton.examples.phoenx.analog_digital_clock_scene import SCENE
from newton.examples.phoenx.example_phoenx_analog_digital_clock import (
    ASSETS,
    _apply_angular_drag,
    _load_mesh,
    _transform,
    build_scene,
)


class TestAnalogDigitalClock(unittest.TestCase):
    def test_joint_frames(self):
        """Apply authored body scales to all 31 hinge anchors."""
        bodies, joints = SCENE["bodies"], SCENE["joints"]
        self.assertEqual(len(bodies), 25)
        self.assertEqual(len(joints), 31)
        self.assertFalse(any(b["kinematic"] for b in bodies))
        for joint in joints:
            frames = []
            for body, values in zip((joint["parent"], joint["child"]), joint["frames"], strict=True):
                frame = _transform(values)
                if body >= 0:
                    frame = _transform(bodies[body]["pose"]) * frame
                frames.append(frame)
            a, b = frames
            self.assertLess(float(wp.length(wp.transform_get_translation(a) - wp.transform_get_translation(b))), 1.0e-6)
            axis = wp.vec3(*{"X": (1, 0, 0), "Y": (0, 1, 0), "Z": (0, 0, 1)}[joint["axis"]])
            self.assertGreater(float(wp.dot(wp.transform_vector(a, axis), wp.transform_vector(b, axis))), 0.99999)

    def test_mesh_normals_and_indices(self):
        """Retain finite unit normals and valid triangles in every standalone OBJ."""
        for filename in sorted({shape["mesh"] for shape in SCENE["shapes"]}):
            with self.subTest(mesh=filename):
                with wp.ScopedDevice("cpu"):
                    mesh = _load_mesh(ASSETS / filename, 0.5)
                self.assertTrue(np.isfinite(mesh.vertices).all())
                self.assertEqual(len(mesh.indices) % 3, 0)
                self.assertGreaterEqual(int(mesh.indices.min()), 0)
                self.assertLess(int(mesh.indices.max()), len(mesh.vertices))
                self.assertIsNotNone(mesh.normals)
                self.assertEqual(mesh.normals.shape, mesh.vertices.shape)
                np.testing.assert_allclose(np.linalg.norm(mesh.normals, axis=1), 1.0, atol=2.0e-5)

    def test_angular_drag_in_rotated_body_frame(self):
        """Authored drag applies world torque using the body's inertia tensor."""
        rotation = wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), np.pi / 2)
        with wp.ScopedDevice("cpu"):
            poses = wp.array([wp.transform(wp.vec3(), rotation)], dtype=wp.transform)
            velocities = wp.array([wp.spatial_vector(0.0, 0.0, 0.0, 1.0, 2.0, 3.0)], dtype=wp.spatial_vector)
            inertia = wp.array([wp.mat33(2.0, 0.0, 0.0, 0.0, 3.0, 0.0, 0.0, 0.0, 4.0)], dtype=wp.mat33)
            damping = wp.array([10.0], dtype=float)
            forces = wp.zeros(1, dtype=wp.spatial_vector)
            wp.launch(_apply_angular_drag, dim=1, inputs=[poses, velocities, inertia, damping], outputs=[forces])
            np.testing.assert_allclose(forces.numpy()[0], [0, 0, 0, -30, -40, -120], atol=3e-5)

    def test_mass_and_inertia(self):
        """Density materials reach the builder and leave every moving part physical."""
        # SDF cooking is unrelated to mass properties and need not run in this test.
        with wp.ScopedDevice("cpu"), patch.object(newton.Mesh, "build_sdf"):
            model = build_scene().finalize(skip_validation_joints=True)
        masses = model.body_mass.numpy()
        self.assertTrue(np.all(masses > 0.0))
        self.assertTrue(np.all(np.linalg.eigvalsh(model.body_inertia.numpy()) > 0.0))
        cam = model.joint_label.index("/World/Clock/Cams/RevoluteJoint")
        dof = model.joint_qd_start.numpy()[cam]
        self.assertEqual(float(model.joint_target_ke.numpy()[dof]), 0.0)
        self.assertEqual(float(model.joint_target_kd.numpy()[dof]), 0.0)
        labels = model.body_label
        follower = masses[labels.index("/World/Clock/Follower/follower_obj0")]
        digit = masses[labels.index("/World/Clock/Digit/DigitHorizontalLow")]
        self.assertAlmostEqual(float(follower), 0.6654444, places=5)
        self.assertAlmostEqual(float(digit), 0.00456177, places=7)

    def test_drives_materials_and_collision_filters(self):
        """Preserve the source springs, velocity drives, densities, and allowed contacts."""
        joints = SCENE["joints"]
        motors = [j for j in joints if j["drive_enabled"] and j["velocity"] != 0.0]
        self.assertEqual(len(motors), 1)
        np.testing.assert_allclose(sorted(j["velocity"] for j in motors), np.deg2rad([100]))
        np.testing.assert_allclose([j["damping"] for j in motors], 10 * 180 / np.pi)
        cam = next(j for j in joints if j["label"] == "/World/Clock/Cams/RevoluteJoint")
        self.assertFalse(cam["drive_enabled"])
        followers = [j for j in joints if "/Follower/" in j["label"]]
        self.assertEqual(len(followers), 7)
        self.assertTrue(all(j["collision_enabled"] and j["drive_enabled"] for j in followers))
        np.testing.assert_allclose([j["target"] for j in followers], np.pi / 4)
        np.testing.assert_allclose([j["stiffness"] for j in followers], 0.03 * 180 / np.pi)
        shapes = SCENE["shapes"]
        self.assertEqual(len(shapes), 54)
        self.assertTrue(all(s["friction"] == 0.0 for s in shapes if s["density"] in (30.0, 5000.0)))
        self.assertTrue(all(s["friction"] == 0.5 for s in shapes if s["density"] == 1000.0))
        self.assertEqual({s["density"] for s in shapes if "/Digit/" in s["label"] and s["collision"]}, {30.0})
        self.assertEqual({s["density"] for s in shapes if "/Follower/" in s["label"]}, {5000.0})
        self.assertAlmostEqual(SCENE["ground_height"], -0.011488295428744681)


if __name__ == "__main__":
    unittest.main()
