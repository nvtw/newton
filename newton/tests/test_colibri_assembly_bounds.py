# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Check Colibri escape diagnostics independently of global assembly motion."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from newton.examples.kamino import example_kamino_colibri as scene
from newton.examples.kamino.example_kamino_colibri import Example
from newton.examples.phoenx.example_phoenx_colibri import Example as PhoenxExample


def _array(value):
    return SimpleNamespace(numpy=lambda: np.asarray(value))


def _example():
    example = Example.__new__(Example)
    example.fix_base = False
    example.initial_q = np.array(
        [[0, 0, 0, 0, 0, 0, 1], [0.2, 0, 0, 0, 0, 0, 1], [0, 1, 0, 0, 0, 0, 1]], dtype=np.float32
    )
    example.model = SimpleNamespace(body_label=["FrameGround", "Frame", "Flower"])
    for name in ("joint_parent", "joint_child", "joint_X_p", "joint_X_c", "joint_type", "joint_axis", "joint_qd_start"):
        setattr(example.model, name, _array([]))
    example.state_0 = SimpleNamespace(body_q=_array(example.initial_q.copy()), body_qd=_array(np.zeros((3, 6))))
    return example


class TestColibriAssemblyBounds(unittest.TestCase):
    def test_phoenx_admission_default(self):
        """Colibri defaults to geometric admission and retains the legacy control."""
        parser = PhoenxExample.create_parser()
        self.assertTrue(parser.parse_args([]).geometric_candidates)
        self.assertEqual(parser.parse_args([]).substeps, 24)
        self.assertEqual(parser.parse_args([]).iterations, 1)
        self.assertFalse(parser.parse_args(["--velocity-filtered-candidates"]).geometric_candidates)

    def test_authored_mesh_colors(self):
        """Preserve white unbound meshes and explicitly colored USD parts."""
        labels = ("Frame/FrameMesh", "FrameGround/Flower/Flower_Stem", "FrameGround/Base")
        shapes = [shape for shape in scene.SHAPES if shape[2] in labels]
        self.assertEqual(len(shapes), len(labels))
        # Substitute small geometry so this color check needs no external assets or SDFs.
        mesh = scene.trimesh.creation.box()
        with (
            patch.object(scene, "SHAPES", shapes),
            patch.object(scene, "COLLISION_LABELS", []),
            patch.object(scene.trimesh, "load", return_value=mesh),
        ):
            builder = scene.build_scene(body_count=2, attach_flower_to_base=True)
        for label in labels:
            expected = (1e-6, 1e-6, 1e-6) if label == "FrameGround/Base" else (1.0, 1.0, 1.0)
            np.testing.assert_allclose(builder.shape_color[builder.shape_label.index(label)], expected)

    def test_counterweight_density_scales_composite_mass_properties(self):
        """Scale only the counterweight contribution, including its parallel-axis moment."""
        shapes = [shape for shape in scene.SHAPES if shape[2] in ("Frame/Cylinder", "Frame/Cylinders/Cylinder")]
        self.assertEqual(len(shapes), 2)
        properties = []
        with patch.object(scene, "SHAPES", shapes):
            for scale in (0.0, 0.3, 1.0):
                builder = scene.build_scene(body_count=2, counterweight_density_scale=scale)
                body = builder.body_label.index("Frame")
                mass = builder.body_mass[body]
                center = np.asarray(builder.body_com[body], dtype=float)
                inertia = np.asarray(builder.body_inertia[body], dtype=float).reshape(3, 3)
                origin_inertia = inertia + mass * (np.dot(center, center) * np.eye(3) - np.outer(center, center))
                properties.append((mass, mass * center, origin_inertia))
        for component in range(3):
            low, scaled, full = (item[component] for item in properties)
            np.testing.assert_allclose(scaled, low + 0.3 * (full - low), rtol=2e-5, atol=1e-10)
        self.assertGreater(properties[0][0], 0.0)
        self.assertGreater(properties[2][0], properties[1][0])
        self.assertEqual(PhoenxExample.create_parser().parse_args([]).counterweight_density_scale, 1.0)

    def test_attached_flower_and_slider_belong_to_base(self):
        """Make flower and helper shapes contribute to the moving base body."""
        flower_shapes = [
            shape
            for shape in scene.SHAPES
            if shape[2]
            in (
                "FrameGround/Flower/Flower_Stem",
                "FrameGround/Flower/Slider/Cube",
            )
        ]
        self.assertEqual(len(flower_shapes), 2)
        # Use primitive stand-ins to isolate body ownership from mesh assets.
        shapes = [
            (name, "cylinder", label, (0, 0, 0, 0, 0, 0, 1), (0.01, 0.02)) for name, _, label, _, _ in flower_shapes
        ]
        with patch.object(scene, "SHAPES", shapes):
            builder = scene.build_scene(body_count=1, attach_flower_to_base=True)
            source = scene.build_scene(body_count=1)
        source_flower = source.body_label.index("Flower")
        self.assertIn("Flower/free", source.joint_label)
        for _, _, label, _, _ in shapes:
            self.assertEqual(source.shape_body[source.shape_label.index(label)], source_flower)
        self.assertNotIn("Flower", builder.body_label)
        self.assertNotIn("Flower/free", builder.joint_label)
        base = builder.body_label.index("FrameGround")
        self.assertGreater(builder.body_mass[base], 0.0)
        for _, _, label, _, _ in shapes:
            self.assertEqual(builder.shape_body[builder.shape_label.index(label)], base)

    def test_phoenx_settled_support_creep(self):
        """Reject support translation and rotation after the settling interval."""
        example = PhoenxExample.__new__(PhoenxExample)
        example._support_test_enabled = True
        example._support_reference = None
        example.model = SimpleNamespace(body_label=["FrameGround"])
        q = np.array([[0, 0, 0, 0, 0, 0, 1]], dtype=float)
        example.state_0 = SimpleNamespace(body_q=_array(q))
        example.sim_time = 1.0
        example._test_support_stationarity()
        self.assertIsNone(example._support_reference)
        example.sim_time = 2.0
        example._test_support_stationarity()
        q[0, 0] = 0.000003
        example._test_support_stationarity()
        q[0, 0] = 0.0001
        with self.assertRaisesRegex(AssertionError, "Support creep"):
            example._test_support_stationarity()
        q[0, 0] = 0
        q[0, 3:] = [0, 0, np.sin(0.0005), np.cos(0.0005)]
        with self.assertRaisesRegex(AssertionError, "Support rotation"):
            example._test_support_stationarity()
        example._support_test_enabled = False
        example._test_support_stationarity()

    def test_phoenx_crank_tracking(self):
        """Accept the measured drive speed and reject the old contact-order deficit."""
        example = PhoenxExample.__new__(PhoenxExample)
        example._support_test_enabled = True
        example._drive_test_time = None
        example._drive_test_duration = 0.0
        example._drive_test_integral = 0.0
        example.model = SimpleNamespace(joint_label=["Frame/Crank"], joint_qd_start=_array([0]))
        example.control = SimpleNamespace(joint_target_qd=_array([-3.4906585]))
        example.state_0 = SimpleNamespace(joint_qd=_array([-3.483]))
        example.sim_time = 2.0
        example._test_drive_tracking()
        example.sim_time = 3.0
        example._test_drive_tracking()
        example.state_0.joint_qd = _array([-3.04])
        example.sim_time = 5.0
        with self.assertRaisesRegex(AssertionError, "Crank tracking"):
            example._test_drive_tracking()

    def test_free_assembly_rigid_motion(self):
        """Global translation and rotation preserve a free assembly's escape result."""
        example = _example()
        q = example.initial_q.copy()
        # Translate two meters and rotate the assembly 90 degrees around Z.
        q[:2, :3] = np.array([[2, -1, 0.4], [2, -0.8, 0.4]])
        q[:2, 3:] = [0, 0, np.sqrt(0.5), np.sqrt(0.5)]
        example.state_0.body_q = _array(q)
        example.test_post_step()

    def test_detached_body_rejected(self):
        """Relative body escape is rejected even if the base itself moves."""
        example = _example()
        q = example.initial_q.copy()
        q[:2, 0] += 2
        q[1, 1] += 0.6
        example.state_0.body_q = _array(q)
        with self.assertRaisesRegex(AssertionError, "escaped"):
            example.test_post_step()

    def test_fixed_base_motion_rejected(self):
        """Fixed-base diagnostics still reject movement smaller than the escape bound."""
        example = _example()
        example.fix_base = True
        q = example.initial_q.copy()
        q[0, 0] = 0.01
        example.state_0.body_q = _array(q)
        with self.assertRaises(AssertionError):
            example.test_post_step()


if __name__ == "__main__":
    unittest.main()
