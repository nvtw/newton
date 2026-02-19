import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from newton._src.viewer.optix.camera import FreeCamera
from newton._src.viewer.optix.mesh import load_obj_mesh
from newton._src.viewer.optix.mini_renderer import (
    build_optix_instance_dtype,
    build_renderer_params_dtype,
    pose7_to_mat4,
)


class TestMiniRendererHelpers(unittest.TestCase):
    def test_renderer_params_dtype_layout(self):
        dt = build_renderer_params_dtype()
        self.assertEqual(dt.itemsize, 128)
        self.assertEqual(dt.fields["image"][1], 0)
        self.assertEqual(dt.fields["trav_handle"][1], 24)
        self.assertEqual(dt.fields["cam_px"][1], 32)
        self.assertEqual(dt.fields["cam_wz"][1], 76)
        self.assertEqual(dt.fields["instance_vertex_ptrs"][1], 80)
        self.assertEqual(dt.fields["instance_index_ptrs"][1], 88)
        self.assertEqual(dt.fields["instance_normal_ptrs"][1], 96)
        self.assertEqual(dt.fields["instance_color_ptrs"][1], 104)

    def test_pose7_to_mat4_identity(self):
        m = pose7_to_mat4([1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 1.0])
        self.assertEqual(m.shape, (4, 4))
        self.assertAlmostEqual(float(m[0, 3]), 1.0)
        self.assertAlmostEqual(float(m[1, 3]), 2.0)
        self.assertAlmostEqual(float(m[2, 3]), 3.0)
        self.assertTrue(np.allclose(m[:3, :3], np.eye(3), atol=1.0e-6))

    def test_optix_instance_dtype_layout(self):
        dt = build_optix_instance_dtype()
        self.assertEqual(dt.itemsize, 80)
        self.assertEqual(dt.fields["transform"][1], 0)
        self.assertEqual(dt.fields["instanceId"][1], 48)
        self.assertEqual(dt.fields["traversableHandle"][1], 64)

    def test_free_camera_basis(self):
        cam = FreeCamera.create_default()
        cam.set_pose([0.0, 0.0, -5.0], [0.0, 0.0, 0.0], [0.0, 1.0, 0.0])
        pos, u, v, w = cam.get_basis(640, 480)
        self.assertEqual(pos.shape, (3,))
        self.assertEqual(u.shape, (3,))
        self.assertEqual(v.shape, (3,))
        self.assertEqual(w.shape, (3,))
        self.assertGreater(float(np.linalg.norm(w)), 0.9)

    def test_load_obj_mesh(self):
        with TemporaryDirectory() as td:
            obj_path = Path(td) / "tri.obj"
            obj_path.write_text(
                "\n".join(
                    [
                        "v 0 0 0",
                        "v 1 0 0",
                        "v 0 1 0",
                        "f 1 2 3",
                    ]
                ),
                encoding="utf-8",
            )
            v, i = load_obj_mesh(obj_path, scale=2.0)
            self.assertEqual(v.shape, (3, 3))
            self.assertEqual(i.shape, (1, 3))
            self.assertTrue(np.allclose(v[1], np.array([2.0, 0.0, 0.0], dtype=np.float32)))


if __name__ == "__main__":
    unittest.main()
