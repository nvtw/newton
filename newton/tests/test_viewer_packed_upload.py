# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Regression coverage for packed instance uploads and viewer cleanup."""

import inspect
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

from newton._src.viewer.gl.opengl import MeshInstancerGL
from newton._src.viewer.viewer_gl import ViewerGL


class TestViewerPackedUpload(unittest.TestCase):
    def test_packed_upload_preserves_instance_count_and_opacity(self):
        for interop in (False, True):
            with self.subTest(interop=interop):
                viewer = Mock()
                viewer.model_changed = True
                viewer.device.is_cuda = True
                viewer._shape_batches_have_transparency = False
                viewer._layer_force_hidden.return_value = False
                viewer._should_show_shape.return_value = True
                viewer._capsule_keys = set()
                host = np.zeros((5, 4, 4), dtype=np.float32)
                viewer._packed_vbo_xforms_host.numpy.return_value = host
                shape = SimpleNamespace(
                    name="transparent_mesh",
                    flags=0,
                    static=False,
                    geo_type=0,
                    colors=object(),
                    materials=object(),
                    opacities=object(),
                    transparent=True,
                    colors_changed=True,
                    opacities_changed=True,
                )
                viewer._packed_groups = [(0, shape, 2, 3)]
                instancer = Mock(spec=MeshInstancerGL)
                instancer._instance_transform_cuda_buffer = object()
                viewer.objects = {shape.name: instancer}
                with (
                    patch("newton._src.viewer.gl.opengl.ENABLE_CUDA_INTEROP", interop),
                    patch("newton._src.viewer.viewer_gl.wp.launch"),
                    patch("newton._src.viewer.viewer_gl.wp.copy"),
                    patch("newton._src.viewer.viewer_gl.wp.synchronize"),
                ):
                    ViewerGL.log_state(viewer, Mock())
                call = (
                    instancer.update_from_packed_cuda.call_args if interop else instancer.update_from_pinned.call_args
                )
                # Bind against the actual renderer API, so wrong positional arguments
                # cannot silently upload an opacity array as the instance count.
                method = MeshInstancerGL.update_from_packed_cuda if interop else MeshInstancerGL.update_from_pinned
                bound = inspect.signature(method).bind(instancer, *call.args, **call.kwargs).arguments
                self.assertEqual(bound["count"], 3)
                self.assertIs(bound["colors"], shape.colors)
                self.assertIs(bound["materials"], shape.materials)
                self.assertIs(bound["opacities"], shape.opacities)
                if interop:
                    self.assertEqual(bound["offset"], 2)
                    self.assertIs(bound["packed_xforms"], viewer._packed_vbo_xforms)
                else:
                    np.testing.assert_array_equal(bound["host_transforms_np"], host[2:5])

    def test_close_releases_image_logger_and_renderer(self):
        viewer = ViewerGL.__new__(ViewerGL)
        viewer._destroy_simulation_stream = Mock()
        viewer._plot_logger = Mock()
        viewer._invalidate_pbo = Mock()
        viewer._image_logger = Mock()
        viewer._destroy_render_geometry = Mock()
        viewer.renderer = Mock()
        viewer.close()
        viewer._image_logger.clear.assert_called_once_with()
        viewer._plot_logger.clear.assert_called_once_with()
        viewer._destroy_render_geometry.assert_called_once_with()
        viewer.renderer.close.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
