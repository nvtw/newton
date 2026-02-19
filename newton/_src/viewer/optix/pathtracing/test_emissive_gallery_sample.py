"""Tests for the translated Emissive Gallery sample."""

import unittest

from newton._src.viewer.optix.pathtracing.camera import Camera
from newton._src.viewer.optix.pathtracing.emissive_gallery_sample import (
    build_emissive_gallery,
    create_emissive_gallery_viewer,
)
from newton._src.viewer.optix.pathtracing.scene import Scene


class TestEmissiveGallerySample(unittest.TestCase):
    """Validate scene translation from EmissiveGallerySample.cs."""

    def test_scene_construction_counts(self):
        scene = Scene(optix_ctx=None)
        build_emissive_gallery(scene)

        # Based on direct translation of EmissiveGallerySample.cs:
        # 37 boxes + 24 spheres, each currently a separate mesh instance.
        self.assertEqual(scene.mesh_count, 61)
        self.assertEqual(scene.instance_count, 61)
        self.assertEqual(scene.materials.count, 35)

    def test_entrypoint_camera_matches_sample(self):
        viewer = create_emissive_gallery_viewer(1280, 720)
        self.assertIsInstance(viewer.camera, Camera)
        self.assertAlmostEqual(float(viewer.camera.position[0]), 0.0, places=5)
        self.assertAlmostEqual(float(viewer.camera.position[1]), 2.8, places=5)
        self.assertAlmostEqual(float(viewer.camera.position[2]), 12.0, places=5)
        self.assertAlmostEqual(float(viewer.camera.target[0]), 0.0, places=5)
        self.assertAlmostEqual(float(viewer.camera.target[1]), 1.2, places=5)
        self.assertAlmostEqual(float(viewer.camera.target[2]), -1.0, places=5)
        self.assertAlmostEqual(viewer.camera.fov, 55.0, places=5)


if __name__ == "__main__":
    unittest.main()
