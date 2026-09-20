# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Verify the bicycle transmission descriptor and locally extracted meshes."""

import unittest

import numpy as np
import warp as wp

from newton.examples.phoenx.bike_transmission_scene import SCENE
from newton.examples.phoenx.example_phoenx_bike_transmission import (
    ALUMINUM_DENSITY,
    ASSETS,
    CHAIN_JOINT_FRICTION,
    DEFAULT_REAR_LOAD_DAMPING,
    DERAILLEUR_DAMPING_SCALE,
    DERAILLEUR_PRELOAD_SCALE,
    DERAILLEUR_SPRING_LABELS,
    STEEL_DENSITY,
    Example,
    _body_density,
    _joint_drive_parameters,
    _load_mesh,
    _transform,
)


class TestBikeTransmission(unittest.TestCase):
    def test_measured_solver_defaults(self):
        """Keep the validated low-work solver configuration."""
        args = Example.create_parser().parse_args([])
        self.assertEqual(args.substeps, 8)
        self.assertEqual(args.iterations, 4)

    def test_derailleur_spring_preload(self):
        """Increase derailleur preload without stiffening its dynamic response."""
        self.assertEqual(DERAILLEUR_PRELOAD_SCALE, 3.0)
        self.assertEqual(DERAILLEUR_DAMPING_SCALE, 2.0)
        for joint in SCENE["joints"]:
            stiffness, damping, target, _ = _joint_drive_parameters(
                joint, DERAILLEUR_PRELOAD_SCALE, DERAILLEUR_DAMPING_SCALE
            )
            self.assertEqual(stiffness, joint["stiffness"])
            expected_damping = (
                joint["damping"] * DERAILLEUR_DAMPING_SCALE
                if joint["label"] in DERAILLEUR_SPRING_LABELS
                else (
                    DEFAULT_REAR_LOAD_DAMPING
                    if joint["label"].endswith("/BackGears/LoadRevoluteJoint")
                    else joint["damping"]
                )
            )
            self.assertEqual(damping, expected_damping)
            expected_target = (
                joint["target"] * DERAILLEUR_PRELOAD_SCALE
                if joint["label"] in DERAILLEUR_SPRING_LABELS
                else joint["target"]
            )
            self.assertAlmostEqual(target, expected_target)

    def test_physical_materials_and_pin_friction(self):
        """Use SI material densities and a small chain-pin Coulomb torque."""
        self.assertEqual(_body_density("/World/Xform/Chain/A0"), STEEL_DENSITY)
        self.assertEqual(_body_density("/World/Xform/FrontGears"), ALUMINUM_DENSITY)
        self.assertEqual(_body_density("/World/Xform/BackGears"), ALUMINUM_DENSITY)
        self.assertEqual(
            _body_density("/World/Xform/Changer/RD_R9250_CAGE/UpperSmallGearCOMPOUND037"),
            ALUMINUM_DENSITY,
        )
        self.assertEqual(CHAIN_JOINT_FRICTION, 5.0e-4)
        self.assertTrue(Example.overlap_simulation_render)

    def test_joint_frames_and_closed_chain(self):
        """Preserve the closed 120-link chain and coincident hinge frames."""
        bodies, joints = SCENE["bodies"], SCENE["joints"]
        self.assertEqual(len(bodies), 134)
        self.assertEqual(len(joints), 134)
        self.assertEqual(sum(b["kinematic"] for b in bodies), 2)
        chain = {i for i, body in enumerate(bodies) if "/Chain/" in body["label"]}
        self.assertEqual(len(chain), 120)
        degree = dict.fromkeys(chain, 0)
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
            if joint["parent"] in chain and joint["child"] in chain:
                degree[joint["parent"]] += 1
                degree[joint["child"]] += 1
        self.assertEqual(set(degree.values()), {2})

    def test_mesh_normals_and_indices(self):
        """Retain finite unit normals and valid triangles in every standalone OBJ."""
        first_mesh = ASSETS / SCENE["shapes"][0]["mesh"]
        if not first_mesh.exists():
            self.skipTest("copyrighted BikeTransmission OBJ meshes are local-only")
        for filename in sorted({shape["mesh"] for shape in SCENE["shapes"]}):
            with self.subTest(mesh=filename):
                mesh = _load_mesh(ASSETS / filename, 0.5)
                self.assertTrue(np.isfinite(mesh.vertices).all())
                self.assertEqual(len(mesh.indices) % 3, 0)
                self.assertGreaterEqual(int(mesh.indices.min()), 0)
                self.assertLess(int(mesh.indices.max()), len(mesh.vertices))
                self.assertIsNotNone(mesh.normals)
                self.assertEqual(mesh.normals.shape, mesh.vertices.shape)
                np.testing.assert_allclose(np.linalg.norm(mesh.normals, axis=1), 1.0, atol=2.0e-5)

    def test_drive_units(self):
        """Convert the source 50-degree-per-second crank drive and angular gains to SI."""
        crank = next(j for j in SCENE["joints"] if j["label"].endswith("/FrontGears/RevoluteJoint"))
        self.assertAlmostEqual(crank["velocity"], -np.deg2rad(50.0))
        self.assertAlmostEqual(crank["damping"], 100.0 * 0.01**2 * 180.0 / np.pi)
        self.assertEqual(SCENE["gravity"], [0, 0, -9.8])


if __name__ == "__main__":
    unittest.main()
