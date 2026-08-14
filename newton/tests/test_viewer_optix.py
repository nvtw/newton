# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import inspect
import unittest
from types import SimpleNamespace

import numpy as np
import warp as wp

import newton
import newton.examples
from newton.viewer import ViewerBase, ViewerGL, ViewerOptix

try:
    import warp_optix
except ImportError:
    warp_optix = None


class _FakeScene:
    def __init__(self):
        self._meshes = []
        self._instances = []

    def set_instance_material_ids_host(self, material_ids):
        del material_ids


class _FakeOptixApi:
    def __init__(self, width: int = 8, height: int = 6):
        self.width = width
        self.height = height
        self.scene = _FakeScene()
        self.dlss_enabled = True
        self.viewer = type("FakePathTracer", (), {"tonemapped_output": None})()
        self.temporal_reset_count = 0
        self.tonemap_exposure = 1.0
        self.tonemap_contrast = 1.0
        self.tonemap_saturation = 1.0
        self.sky_parameters = None

    def initialize(self):
        return True

    def set_use_procedural_sky(self, enabled):
        del enabled

    def set_sky_parameters(self, **kwargs):
        self.sky_parameters = kwargs

    def set_camera_look_at(self, position, target, up, fov):
        del position, target, up, fov

    def get_frame_uint8(self):
        return np.full((self.height, self.width, 4), 127, dtype=np.uint8)

    def reset_temporal_history(self):
        self.temporal_reset_count += 1

    def clear_scene(self):
        self.scene = _FakeScene()

    def close(self):
        return


class TestViewerOptix(unittest.TestCase):
    def test_authored_mesh_material_detection(self):
        """Apply fallback materials only to meshes without authored PBR data."""
        model = SimpleNamespace(
            shape_source=[
                SimpleNamespace(roughness=None, metallic=None, texture=None),
                SimpleNamespace(roughness=0.2, metallic=None, texture=None),
            ]
        )
        batch = SimpleNamespace(geo_type=newton.GeoType.MESH, model_shapes=[0])
        self.assertFalse(ViewerOptix._has_authored_mesh_material(model, batch))

        batch.model_shapes = [1]
        self.assertTrue(ViewerOptix._has_authored_mesh_material(model, batch))

    def test_simulation_render_overlap_disabled(self):
        """Keep OptiX simulation and rendering serialized for stable scene updates."""
        viewer = ViewerOptix.__new__(ViewerOptix)
        self.assertFalse(viewer.supports_simulation_render_overlap)

    def test_public_viewer_and_example_option(self):
        """Expose ViewerOptix through the public API and example parser."""
        self.assertTrue(issubclass(ViewerOptix, ViewerBase))
        parser = newton.examples.create_parser()
        viewer_action = next(action for action in parser._actions if action.dest == "viewer")
        self.assertIn("optix", viewer_action.choices)
        gl_parameters = inspect.signature(ViewerGL).parameters
        args = parser.parse_args([])
        self.assertEqual(args.optix_dlss_quality, "performance")
        self.assertEqual(args.optix_max_bounces, 3)
        self.assertEqual(args.optix_direct_light_samples, 1)
        self.assertEqual(args.optix_samples_per_frame, 1)
        optix_parameters = inspect.signature(ViewerOptix).parameters
        self.assertEqual(optix_parameters["width"].default, gl_parameters["width"].default)
        self.assertEqual(optix_parameters["height"].default, gl_parameters["height"].default)
        self.assertEqual(optix_parameters["max_instances"].default, 16384)
        self.assertEqual(optix_parameters["dlss_quality"].default, "performance")
        self.assertEqual(optix_parameters["max_bounces"].default, 3)
        self.assertEqual(optix_parameters["direct_light_samples"].default, 1)
        self.assertEqual(optix_parameters["samples_per_frame"].default, 1)
        self.assertEqual(optix_parameters["ground_checker_size"].default, 1.0)

    def test_ground_checker_subdivisions_use_metric_plane_extents(self):
        """Size plane checker subdivisions in meters and support disabling them."""
        viewer = ViewerOptix.__new__(ViewerOptix)
        viewer._ground_checker_size = 1.0
        viewer._mesh_ids = {"ground": 0}
        viewer._api = SimpleNamespace(
            scene=SimpleNamespace(
                _meshes=[
                    SimpleNamespace(
                        vertices=np.asarray(
                            ((-5.0, -3.0, 0.0), (5.0, -3.0, 0.0), (5.0, 3.0, 0.0), (-5.0, 3.0, 0.0)),
                            dtype=np.float32,
                        )
                    )
                ]
            )
        )

        self.assertEqual(viewer._checker_subdivisions_for_mesh("ground"), (10.0, 6.0))
        viewer._ground_checker_size = None
        self.assertEqual(viewer._checker_subdivisions_for_mesh("ground"), (0.0, 0.0))

    @unittest.skipIf(warp_optix is None, "warp_optix is not installed")
    def test_set_camera_updates_backend_pose(self):
        """Keep the configured camera pose through the first input update."""
        api = _FakeOptixApi()
        viewer = ViewerOptix(device="cpu", headless=True, enable_imgui=False, api=api)
        try:
            viewer.set_camera(wp.vec3(1.2, 0.75, 0.4), pitch=-12.0, yaw=180.0)
            viewer._update_camera_from_input(0.0)

            np.testing.assert_allclose(viewer._camera_position, (1.2, 0.75, 0.4))
            np.testing.assert_allclose(np.asarray(viewer.camera.pos), (1.2, 0.75, 0.4))
            self.assertAlmostEqual(viewer._camera_pitch, -12.0)
            self.assertAlmostEqual(viewer._camera_yaw, -180.0)
        finally:
            viewer.close()

    @unittest.skipIf(warp_optix is None, "warp_optix is not installed")
    def test_default_color_palette(self):
        """Remap automatic colors while preserving explicit and authored colors."""
        api = _FakeOptixApi()
        viewer = ViewerOptix(device="cpu", headless=True, enable_imgui=False, api=api)
        try:
            default_colors = np.asarray([ViewerBase._shape_color_map(i) for i in range(3)], dtype=np.float32)
            explicit_color = np.array((0.21, 0.43, 0.65), dtype=np.float32)
            default_colors[1] = explicit_color
            viewer.model = SimpleNamespace(shape_source=[None, None, SimpleNamespace(color=tuple(default_colors[2]))])
            viewer._optix_model_shape_batches["shapes"] = SimpleNamespace(model_shapes=[0, 1, 2])

            mapped = viewer._palette_colors("shapes", wp.array(default_colors, dtype=wp.vec3, device="cpu"))
            mapped_numpy = mapped.numpy()
            np.testing.assert_allclose(mapped_numpy[0], ViewerOptix._DEFAULT_COLOR_PALETTE[0])
            np.testing.assert_allclose(mapped_numpy[1], explicit_color)
            np.testing.assert_allclose(mapped_numpy[2], default_colors[2])

            viewer.set_default_color_palette(((0.9, 0.3, 0.0),))
            mapped = viewer._palette_colors("shapes", wp.array(default_colors, dtype=wp.vec3, device="cpu"))
            np.testing.assert_allclose(mapped.numpy()[0], (0.9, 0.3, 0.0))
            with self.assertRaises(ValueError):
                viewer.set_default_color_palette(())
        finally:
            viewer.close()

    @unittest.skipIf(warp_optix is None, "warp_optix is not installed")
    def test_time_of_day_updates_sky(self):
        """Move the procedural sun and reset temporal history when time changes."""
        api = _FakeOptixApi()
        viewer = ViewerOptix(device="cpu", headless=True, enable_imgui=False, api=api)
        try:
            self.assertAlmostEqual(viewer.time_of_day, 12.0)
            self.assertAlmostEqual(viewer.sky_intensity, 1.0)
            self.assertFalse(viewer.grayscale_sky)
            self.assertIsNotNone(api.sky_parameters)
            self.assertEqual(api.temporal_reset_count, 0)
            np.testing.assert_allclose(api.sky_parameters["ground_color"], (0.4, 0.4, 0.4), atol=1.0e-5)
            self.assertAlmostEqual(api.sky_parameters["sun_glow_intensity"], 1.0)
            self.assertFalse(api.sky_parameters["grayscale"])

            viewer.grayscale_sky = True
            self.assertEqual(api.temporal_reset_count, 1)
            self.assertTrue(api.sky_parameters["grayscale"])
            viewer.sky_intensity = 1.5
            self.assertEqual(api.temporal_reset_count, 2)
            self.assertAlmostEqual(api.sky_parameters["multiplier"], 1.5)
            viewer.time_of_day = 18.0
            self.assertEqual(api.temporal_reset_count, 3)
            np.testing.assert_allclose(api.sky_parameters["sun_direction"], (1.0, 0.0, 0.0), atol=1.0e-6)
            with self.assertRaises(ValueError):
                viewer.time_of_day = 25.0
            with self.assertRaises(ValueError):
                viewer.sky_intensity = -1.0
        finally:
            viewer.close()

    @unittest.skipIf(warp_optix is not None, "Exercise the missing-dependency path only without warp_optix")
    def test_missing_backend_error(self):
        """Report how to install the optional backend when it is unavailable."""
        with self.assertRaisesRegex(ImportError, "otk-pyoptix"):
            ViewerOptix(headless=True)

    @unittest.skipIf(warp_optix is None, "warp_optix is not installed")
    def test_pause_step_and_frame_extraction(self):
        """Match ViewerGL pause, single-step, and frame-extraction behavior."""
        api = _FakeOptixApi()
        viewer = ViewerOptix(
            width=api.width,
            height=api.height,
            device="cpu",
            headless=True,
            paused=True,
            enable_imgui=False,
            api=api,
        )
        try:
            self.assertFalse(viewer.should_step())
            viewer._step_requested = True
            self.assertTrue(viewer.should_step())
            self.assertFalse(viewer.should_step())

            reset_calls = []
            viewer.set_reset_callback(lambda: reset_calls.append(True))
            viewer._reset_callback()
            self.assertAlmostEqual(viewer.exposure, 0.68)
            self.assertEqual(viewer._ground_color, (0.7, 0.7, 0.7))
            self.assertAlmostEqual(viewer._ground_roughness, 0.8)
            self.assertAlmostEqual(viewer._ground_checker_size, 1.0)
            self.assertAlmostEqual(viewer._default_roughness, 0.42)
            self.assertAlmostEqual(viewer._default_ior, 1.46)
            self.assertAlmostEqual(viewer._default_specular, 0.75)
            self.assertAlmostEqual(viewer._default_clearcoat, 0.03)
            self.assertAlmostEqual(viewer._default_clearcoat_roughness, 0.4)
            self.assertAlmostEqual(viewer.tonemap_saturation, 1.1)
            self.assertAlmostEqual(viewer.tonemap_contrast, 1.08)
            self.assertEqual(reset_calls, [True])
            self.assertEqual(api.temporal_reset_count, 1)

            frame = viewer.get_frame()
            self.assertEqual(frame.shape, (api.height, api.width, 3))
            self.assertEqual(frame.dtype, wp.uint8)
            np.testing.assert_array_equal(frame.numpy(), 127)
        finally:
            viewer.close()


if __name__ == "__main__":
    unittest.main(verbosity=2)
