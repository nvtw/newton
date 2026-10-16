# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import importlib.util
import unittest

import numpy as np
import warp as wp

import newton
from newton.viewer import ViewerOptix


@wp.kernel
def _translate_body(poses: wp.array[wp.transform]):
    pose = poses[0]
    poses[0] = wp.transform(
        wp.transform_get_translation(pose) + wp.vec3(0.25, 0.0, 0.0), wp.transform_get_rotation(pose)
    )


@unittest.skipUnless(importlib.util.find_spec("warp_optix") is not None, "Requires warp-optix")
class TestViewerOptix(unittest.TestCase):
    def test_ground_checker_survives_color_updates(self):
        """Preserve rendered checker contrast through repeated model color updates."""
        if not wp.is_cuda_available():
            self.skipTest("Requires CUDA")
        with wp.ScopedDevice("cuda:0"):
            viewer = ViewerOptix(
                width=240,
                height=160,
                headless=True,
                enable_imgui=False,
                enable_dlss_rr=False,
                samples_per_frame=8,
                max_bounces=1,
            )
            self.addCleanup(viewer.close)
            builder = newton.ModelBuilder()
            builder.add_ground_plane()
            model = builder.finalize()
            state = model.state()
            viewer.set_model(model)
            viewer.set_camera(wp.vec3(0.0, 0.0, 5.0), pitch=-90.0, yaw=0.0)
            contrasts = []
            for frame in range(4):
                viewer.begin_frame(frame / 60.0)
                viewer.log_state(state)
                viewer.end_frame()
                pixels = viewer.get_frame().numpy()
                self.assertEqual(pixels.shape, (160, 240, 3))
                contrasts.append(float(np.std(pixels[80, :, :].mean(axis=1))))
                ground = next(
                    shapes for shapes in viewer._shape_instances.values() if shapes.geo_type == newton.GeoType.PLANE
                )
                properties = viewer._device_material_batches[ground.name][2].numpy()
                np.testing.assert_allclose(properties[:, 2:], ((1000.0, 1000.0),))
            self.assertGreater(contrasts[0], 2.0)
            self.assertGreater(contrasts[-1], 0.75 * contrasts[0])

    def test_overlap_preserves_render_snapshots(self):
        """Render stable buffered poses while another CUDA stream updates the model."""
        if not wp.is_cuda_available():
            self.skipTest("Requires CUDA")
        with wp.ScopedDevice("cuda:0"):
            viewer = ViewerOptix(width=160, height=120, headless=True, enable_imgui=False, enable_dlss_rr=False)
            self.addCleanup(viewer.close)
            if not viewer.supports_simulation_render_overlap:
                self.skipTest("Requires nonblocking CUDA streams")
            builder = newton.ModelBuilder()
            body = builder.add_body(xform=wp.transform((0.0, 0.0, 0.5), wp.quat_identity()))
            builder.add_shape_box(body, hx=0.25, hy=0.25, hz=0.25)
            builder.add_ground_plane()
            model = builder.finalize()
            state = model.state()
            snapshots = (model.state(), model.state())
            done = (wp.Event(model.device), wp.Event(model.device))
            viewer.set_model(model)
            viewer.set_camera(wp.vec3(2.0, -4.0, 2.0), pitch=-15.0, yaw=116.0)
            for frame in range(4):
                index = frame % 2

                def snapshot(index=index):
                    wp.wait_event(done[index])
                    wp.copy(snapshots[index].body_q, state.body_q)

                def advance():
                    wp.launch(_translate_body, dim=1, inputs=[state.body_q])

                viewer.launch_simulation_step(advance, prepare_render_state=snapshot)
                viewer.begin_frame(frame / 60.0)
                viewer.log_state(snapshots[index])
                viewer.end_frame()
                wp.record_event(done[index])
                viewer.synchronize_simulation_step()
                self.assertAlmostEqual(float(snapshots[index].body_q.numpy()[body, 0]), 0.25 * frame)
                self.assertAlmostEqual(float(state.body_q.numpy()[body, 0]), 0.25 * (frame + 1))


if __name__ == "__main__":
    unittest.main()
