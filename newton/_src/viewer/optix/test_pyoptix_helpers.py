import unittest

import numpy as np

from newton._src.viewer.optix.example_pyoptix_empty_buffer import _load_native_section
from newton._src.viewer.optix.handles import HandleBuffer
from newton._src.viewer.optix.launch_params import build_launch_params_dtype, pack_launch_params_bytes
from newton._src.viewer.optix.mesh import create_cube_mesh, create_plane_mesh
from newton._src.viewer.optix.scene_core import SceneCore, SceneState


class TestPyoptixHelpers(unittest.TestCase):
    def test_handle_buffer_add_get_remove(self):
        hb = HandleBuffer[int]()
        h0 = hb.add(10)
        h1 = hb.add(20)
        self.assertTrue(h0.is_valid())
        self.assertTrue(h1.is_valid())
        self.assertEqual(hb.get_value(h0), 10)
        self.assertEqual(hb.get_value(h1), 20)
        hb.remove_value(h0)
        ok, value = hb.try_get_value(h0)
        self.assertFalse(ok)
        self.assertIsNone(value)
        self.assertEqual(hb.count, 1)

    def test_launch_params_dtype_offsets(self):
        dt = build_launch_params_dtype()
        self.assertEqual(dt.itemsize, 32)
        self.assertEqual(dt.fields["image"][1], 0)
        self.assertEqual(dt.fields["width"][1], 8)
        self.assertEqual(dt.fields["height"][1], 12)
        self.assertEqual(dt.fields["time_sec"][1], 16)
        self.assertEqual(dt.fields["trav_handle"][1], 24)

    def test_pack_launch_params_bytes_shape(self):
        params, raw = pack_launch_params_bytes(1234, 64, 32, 1.25, 999)
        self.assertEqual(params.shape[0], 1)
        self.assertEqual(raw.dtype, np.uint8)
        self.assertEqual(raw.size, 32)

    def test_mesh_generators(self):
        v, i = create_cube_mesh(2.0)
        self.assertEqual(v.shape, (8, 3))
        self.assertEqual(i.shape, (12, 3))
        pv, pi = create_plane_mesh(4.0, 2.0)
        self.assertEqual(pv.shape, (4, 3))
        self.assertEqual(pi.shape, (2, 3))

    def test_scene_core_state_transitions(self):
        scene = SceneCore()
        mesh_h = scene.add_cube(1.0)
        self.assertTrue(scene.state & SceneState.MESHES_CHANGED)
        self.assertTrue(mesh_h.is_valid())
        t = np.eye(4, dtype=np.float32)
        inst_h = scene.add_instance(t, mesh_h, hit_kernel_sbt_offset=0)
        self.assertTrue(scene.state & SceneState.INSTANCES_CHANGED)
        scene.set_instance_transform(inst_h, np.eye(4, dtype=np.float32))
        self.assertTrue(scene.state & SceneState.INSTANCE_TRANSFORMS_CHANGED)
        self.assertEqual(scene.get_instance_count(), 1)

    def test_native_sections_load(self):
        launch = _load_native_section("launch_params")
        trace = _load_native_section("trace_mode_trace0")
        self.assertIn("struct LaunchParams", launch)
        self.assertIn("__raygen__cube", trace)
        self.assertIn("optixTrace", trace)


if __name__ == "__main__":
    unittest.main()
