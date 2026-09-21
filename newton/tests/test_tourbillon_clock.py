# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Verify the standalone Tourbillon clock scene and its authored joints."""

import unittest
from unittest.mock import patch

import numpy as np
import warp as wp

import newton
from newton.examples.phoenx.example_phoenx_tourbillon_clock import (
    ASSETS,
    BEARING_FRICTION,
    OSCILLATOR_FRICTION_SCALE,
    REDUNDANT_BEARING_JOINT,
    Example,
    _active_joints,
    _load_mesh,
    _transform,
    build_scene,
)
from newton.examples.phoenx.tourbillon_clock_scene import SCENE

HAS_LOCAL_ASSETS = any(ASSETS.glob("*.obj"))


class TestTourbillonClock(unittest.TestCase):
    def test_joint_projection_defaults(self):
        """Project bearings after every grouped contact sweep."""
        args = Example.create_parser().parse_args([])
        self.assertEqual(Example.color_group_size, 2)
        self.assertEqual(args.contact_chunk_size, 64)
        self.assertEqual(args.iterations, 4)
        self.assertEqual(args.direct_joint_projection_passes, args.iterations)
        self.assertEqual(args.substeps, 4)

    def test_scene_and_joint_frames(self):
        """Retain the complete SI mechanism and coincident hinge frames."""
        bodies, joints = SCENE["bodies"], SCENE["joints"]
        self.assertEqual(len(bodies), 24)
        self.assertEqual(len(SCENE["shapes"]), 53)
        self.assertEqual(len(joints), 24)
        self.assertEqual(sum(body["kinematic"] for body in bodies), 1)
        self.assertEqual(len(_active_joints()), 23)
        self.assertNotIn(REDUNDANT_BEARING_JOINT, {joint["label"] for joint in _active_joints()})

        for joint in joints:
            frames = []
            for body, values in zip((joint["parent"], joint["child"]), joint["frames"], strict=True):
                frame = _transform(values)
                if body >= 0:
                    frame = _transform(bodies[body]["pose"]) * frame
                frames.append(frame)
            parent, child = frames
            gap = wp.length(wp.transform_get_translation(parent) - wp.transform_get_translation(child))
            self.assertLess(float(gap), 1.0e-6)
            axis = wp.vec3(*{"X": (1, 0, 0), "Y": (0, 1, 0), "Z": (0, 0, 1)}[joint["axis"]])
            alignment = wp.dot(wp.transform_vector(parent, axis), wp.transform_vector(child, axis))
            self.assertGreater(float(alignment), 0.99999)

    def test_drives_and_materials(self):
        """Preserve the source escapement motor, springs, masses, and friction."""
        joints = {joint["label"]: joint for joint in SCENE["joints"]}
        motor = joints["/World/Clock/TourbillonSpecialGear/RevoluteJoint"]
        self.assertTrue(motor["drive_enabled"])
        self.assertAlmostEqual(motor["velocity"], -np.deg2rad(800.0))
        self.assertAlmostEqual(motor["damping"], 3.0 * 0.01**2 * 180.0 / np.pi)

        oscillator = joints["/World/Clock/TourbillonOscillator/RevoluteJoint_01"]
        self.assertAlmostEqual(oscillator["stiffness"], 5.0 * 0.01**2 * 180.0 / np.pi)
        stopper = joints["/World/Clock/Stopper/RevoluteJoint"]
        self.assertAlmostEqual(stopper["target"], -np.deg2rad(20.0))
        self.assertAlmostEqual(stopper["stiffness"], 10.0 * 0.01**2 * 180.0 / np.pi)
        self.assertAlmostEqual(stopper["damping"], 1.0 * 0.01**2 * 180.0 / np.pi)

        masses = {body["label"]: body["mass"] for body in SCENE["bodies"] if body["mass"] is not None}
        self.assertEqual(masses["/World/Clock/TourbillonFrame"], 1.0)
        self.assertAlmostEqual(masses["/World/Clock/TourbillonAmplifierGear"], 0.1)
        self.assertAlmostEqual(masses["/World/Clock/TourbillonStopper"], 0.1)
        self.assertEqual({shape["density"] for shape in SCENE["shapes"] if shape["collision"]}, {1000.0})

    @unittest.skipUnless(HAS_LOCAL_ASSETS, "copyrighted Tourbillon OBJ assets are installed locally")
    def test_mesh_normals_and_indices(self):
        """Retain finite unit normals and valid triangles in each extracted OBJ."""
        for filename in sorted({shape["mesh"] for shape in SCENE["shapes"]}):
            with self.subTest(mesh=filename), wp.ScopedDevice("cpu"):
                mesh = _load_mesh(ASSETS / filename, 0.5)
                self.assertTrue(np.isfinite(mesh.vertices).all())
                self.assertEqual(len(mesh.indices) % 3, 0)
                self.assertGreaterEqual(int(mesh.indices.min()), 0)
                self.assertLess(int(mesh.indices.max()), len(mesh.vertices))
                self.assertIsNotNone(mesh.normals)
                self.assertEqual(mesh.normals.shape, mesh.vertices.shape)
                np.testing.assert_allclose(np.linalg.norm(mesh.normals, axis=1), 1.0, atol=2.0e-5)

    @unittest.skipUnless(HAS_LOCAL_ASSETS, "copyrighted Tourbillon OBJ assets are installed locally")
    def test_mass_and_inertia(self):
        """Build all rigid bodies with positive mass and inertia."""
        with wp.ScopedDevice("cpu"), patch.object(newton.Mesh, "build_sdf"):
            model = build_scene().finalize(skip_validation_joints=True)
        masses = model.body_mass.numpy()
        moving = np.array([not body["kinematic"] for body in SCENE["bodies"]])
        self.assertTrue(np.all(masses[moving] > 0.0))
        self.assertTrue(np.all(np.linalg.eigvalsh(model.body_inertia.numpy()[moving]) > 0.0))
        self.assertEqual(model.joint_count, 23)
        friction = model.joint_friction.numpy()
        oscillator_joint = model.joint_label.index("/World/Clock/TourbillonOscillator/RevoluteJoint_01")
        oscillator_dof = model.joint_qd_start.numpy()[oscillator_joint]
        expected_friction = np.full_like(friction, BEARING_FRICTION)
        expected_friction[oscillator_dof] *= OSCILLATOR_FRICTION_SCALE
        np.testing.assert_allclose(friction, expected_friction)
        labels = model.body_label
        self.assertAlmostEqual(float(masses[labels.index("/World/Clock/TourbillonFrame")]), 1.0)
        self.assertAlmostEqual(float(masses[labels.index("/World/Clock/TourbillonAmplifierGear")]), 0.1)
        self.assertAlmostEqual(float(masses[labels.index("/World/Clock/TourbillonStopper")]), 0.1)


if __name__ == "__main__":
    unittest.main()
