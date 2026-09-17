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

    def test_render_snapshot_is_independent_of_live_state(self):
        """Render the copied state and its timestamp while physics advances."""
        example = PhoenxExample.__new__(PhoenxExample)

        def state():
            result = scene.newton.State()
            result.body_q = scene.wp.array([[0, 0, 0, 0, 0, 0, 1]], dtype=scene.wp.transform, device="cpu")
            return result

        example.state_0 = state()
        example.contacts = object()
        example._render_states = (state(), state())
        example._render_state_index = 0
        example._render_contacts = [None, None]
        example._render_state_done = (None, None)
        example.sim_time = 0.25
        example.viewer = SimpleNamespace(show_contacts=False)
        example.prepare_render_state()
        snapshot = example._render_states[example._render_state_index]
        example.state_0.body_q.assign([[1, 0, 0, 0, 0, 0, 1]])
        np.testing.assert_array_equal(snapshot.body_q.numpy()[0, :3], [0, 0, 0])
        calls = []
        example.viewer = SimpleNamespace(
            show_contacts=False,
            log_contacts=lambda contacts, state: None,
            begin_frame=calls.append,
            log_state=calls.append,
            end_frame=lambda: None,
        )
        example.sim_time = 0.5
        example.render()
        self.assertEqual(calls[0], 0.25)
        self.assertIs(calls[1], snapshot)
        self.assertFalse(example._render_state_prepared)
        example.render()
        self.assertEqual(calls[2], 0.5)
        self.assertIs(calls[3], example.state_0)

    def test_render_snapshot_waits_only_for_its_own_reader(self):
        """Two buffers wait for their own reuse events, not the other frame."""
        calls = []
        example = PhoenxExample.__new__(PhoenxExample)
        example._render_states = tuple(
            SimpleNamespace(assign=lambda state, index=index: calls.append(("copy", index))) for index in range(2)
        )
        example._render_state_done = (object(), object())
        example._render_state_index = 0
        example._render_contacts = [None, None]
        example.state_0 = object()
        example.contacts = object()
        example.sim_time = 0.0
        example.viewer = SimpleNamespace(
            show_contacts=False,
            log_contacts=lambda contacts, state: None,
            begin_frame=lambda time: None,
            log_state=lambda state: None,
            end_frame=lambda: calls.append(("render", None)),
        )
        with (
            patch.object(scene.wp, "wait_event", side_effect=lambda event: calls.append(("wait", event))),
            patch.object(scene.wp, "record_event", side_effect=lambda event: calls.append(("release", event))),
        ):
            for _ in range(3):
                example.prepare_render_state()
                example.render()
        expected = []
        for index in (1, 0, 1):
            event = example._render_state_done[index]
            expected.extend([("wait", event), ("copy", index), ("release", event), ("render", None)])
        self.assertEqual(calls, expected)

    def test_contact_display_snapshots_only_when_enabled(self):
        """Toggle contact display without racing live buffers or copying when hidden."""
        example = PhoenxExample.__new__(PhoenxExample)
        example.contacts = scene.newton.Contacts(2, 0, device="cpu", requested_attributes={"force"})
        example.contacts.rigid_contact_count.assign([1])
        example.contacts.rigid_contact_point0.assign([[1, 2, 3], [0, 0, 0]])
        example.contacts.force.assign(np.ones((2, 6), dtype=np.float32))
        example.state_0 = scene.newton.State()
        example._render_states = (scene.newton.State(), scene.newton.State())
        example._render_state_index = 0
        example._render_contacts = [None, None]
        example._render_state_done = (None, None)
        example.sim_time = 0.25
        calls = []
        example.viewer = SimpleNamespace(
            show_contacts=False,
            begin_frame=lambda time: calls.append(("time", time)),
            log_state=lambda state: calls.append(("state", state)),
            log_contacts=lambda contacts, state: calls.append(("contacts", contacts, state)),
            end_frame=lambda: None,
        )
        example.prepare_render_state()
        self.assertEqual(example._render_contacts, [None, None])
        example.render()
        example.viewer.show_contacts = True
        for _ in range(3):
            example.prepare_render_state()
            index = example._render_state_index
            snapshot = example._render_contacts[index]
            self.assertIsNotNone(snapshot)
            np.testing.assert_array_equal(snapshot.rigid_contact_count.numpy(), [1])
            np.testing.assert_array_equal(snapshot.rigid_contact_point0.numpy()[0], [1, 2, 3])
            np.testing.assert_array_equal(snapshot.force.numpy(), 1)
            example.contacts.rigid_contact_count.zero_()
            example.contacts.rigid_contact_point0.zero_()
            example.contacts.force.zero_()
            calls.clear()
            example.sim_time = 0.5
            example.render()
            self.assertIn(("contacts", snapshot, example._render_states[index]), calls)
            np.testing.assert_array_equal(snapshot.rigid_contact_count.numpy(), [1])
            np.testing.assert_array_equal(snapshot.rigid_contact_point0.numpy()[0], [1, 2, 3])
            np.testing.assert_array_equal(snapshot.force.numpy(), 1)
            example.contacts.rigid_contact_count.assign([1])
            example.contacts.rigid_contact_point0.assign([[1, 2, 3], [0, 0, 0]])
            example.contacts.force.assign(np.ones((2, 6), dtype=np.float32))
        buffers = tuple(example._render_contacts)
        example.viewer.show_contacts = False
        with patch.object(scene.wp, "copy", side_effect=AssertionError("Unexpected contact copy")):
            example.prepare_render_state()
        example.render()
        self.assertEqual(tuple(example._render_contacts), buffers)
        # Paused and non-overlapping rendering uses matching live data.
        example.viewer.show_contacts = True
        calls.clear()
        example.render()
        self.assertIn(("contacts", example.contacts, example.state_0), calls)

    def test_motor_off_removes_drive_and_servo_damping(self):
        """A zero speed target must not be mistaken for a disabled motor."""
        with patch.object(scene, "SHAPES", []):
            builder = scene.build_scene(enable_frame_drive=False, enable_crank_drive=False)
        self.assertTrue(all(mode == scene.newton.JointTargetMode.NONE for mode in builder.joint_target_mode))
        np.testing.assert_array_equal(builder.joint_target_ke, 0)
        np.testing.assert_array_equal(builder.joint_target_kd, 0)
        self.assertFalse(PhoenxExample.create_parser().parse_args([]).motor_off)
        self.assertTrue(PhoenxExample.create_parser().parse_args(["--motor-off"]).motor_off)

    def test_phoenx_settled_support_creep(self):
        """Reject support translation and rotation after the settling interval."""
        example = PhoenxExample.__new__(PhoenxExample)
        example._support_test_enabled = True
        example._support_reference = None
        example.motor_enabled = False
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
        example.motor_enabled = True
        example._test_support_stationarity()
        example.motor_enabled = False
        example._support_test_enabled = False
        example._test_support_stationarity()

    def test_phoenx_crank_tracking(self):
        """Accept the measured drive speed and reject the old contact-order deficit."""
        example = PhoenxExample.__new__(PhoenxExample)
        example._support_test_enabled = True
        example.motor_enabled = True
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
