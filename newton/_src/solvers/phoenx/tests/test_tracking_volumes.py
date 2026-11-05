# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Tracking volumes: shapes that report exact overlaps without a contact response."""

from __future__ import annotations

import unittest

import numpy as np
import warp as wp

import newton

_HALF = 0.1
_VOLUME_CENTER = 1.0
_VOLUME_HALF = 0.4


@wp.kernel
def _record_overlap_kernel(
    count: wp.array[wp.int32],
    shape0: wp.array[wp.int32],
    shape1: wp.array[wp.int32],
    point0: wp.array[wp.vec3],
    point1: wp.array[wp.vec3],
    normal: wp.array[wp.vec3],
    margin0: wp.array[wp.float32],
    margin1: wp.array[wp.float32],
    shape_body: wp.array[wp.int32],
    body_q: wp.array[wp.transform],
    volume: wp.array[wp.int32],
    overlapping: wp.array[wp.int32],
):
    """Flag an exact overlap: a volume contact whose separation is not positive."""
    k = wp.tid()
    if k >= count[0]:
        return
    a = shape0[k]
    b = shape1[k]
    if volume[a] == 0 and volume[b] == 0:
        return
    p0 = point0[k]
    if shape_body[a] >= 0:
        p0 = wp.transform_point(body_q[shape_body[a]], p0)
    p1 = point1[k]
    if shape_body[b] >= 0:
        p1 = wp.transform_point(body_q[shape_body[b]], p1)
    if wp.dot(p1 - p0, normal[k]) - margin0[k] - margin1[k] <= 0.0:
        overlapping[0] = 1


@unittest.skipUnless(wp.is_cuda_available(), "tracking volume tests use CUDA graph capture")
class TestTrackingVolumes(unittest.TestCase):
    def test_box_falls_through_volume_and_overlap_is_exact(self):
        builder = newton.ModelBuilder()
        builder.add_ground_plane()
        volume = builder.add_shape_box(
            -1, xform=wp.transform((0.0, 0.0, _VOLUME_CENTER), wp.quat_identity()), hx=0.4, hy=0.4, hz=_VOLUME_HALF
        )
        body = builder.add_body(xform=wp.transform((0.0, 0.0, 2.0), wp.quat_identity()))
        builder.add_shape_box(body, hx=_HALF, hy=_HALF, hz=_HALF)
        model = builder.finalize()
        solver = newton.solvers.SolverPhoenX(model, step_layout="single_world", tracking_volumes=[volume])
        state_0, state_1 = model.state(), model.state()
        contacts = model._collision_pipeline.contacts()
        overlapping = wp.zeros(1, dtype=wp.int32)

        with wp.ScopedCapture() as capture:
            overlapping.zero_()
            solver.collide(state_0, contacts)
            wp.launch(
                _record_overlap_kernel,
                dim=contacts.rigid_contact_max,
                inputs=[
                    contacts.rigid_contact_count,
                    contacts.rigid_contact_shape0,
                    contacts.rigid_contact_shape1,
                    contacts.rigid_contact_point0,
                    contacts.rigid_contact_point1,
                    contacts.rigid_contact_normal,
                    contacts.rigid_contact_margin0,
                    contacts.rigid_contact_margin1,
                    model.shape_body,
                    state_0.body_q,
                    solver.world.shape_no_response,
                    overlapping,
                ],
            )
            solver.step(state_0, state_1, None, contacts, 1.0 / 60.0)
            wp.copy(state_0.body_q, state_1.body_q)
            wp.copy(state_0.body_qd, state_1.body_qd)

        frames_inside = 0
        for _ in range(90):
            z = float(state_0.body_q.numpy()[body][2])  # pose the next collide sees
            wp.capture_launch(capture.graph)
            # Exact: overlapping exactly while the box's extent intersects the volume, with a
            # 1 mm tolerance for the pose change between collide and step.
            gap = abs(z - _VOLUME_CENTER) - (_VOLUME_HALF + _HALF)
            if abs(gap) > 1.0e-3:
                self.assertEqual(int(overlapping.numpy()[0]), int(gap < 0.0), f"z={z:.4f}")
            frames_inside += int(overlapping.numpy()[0])
        self.assertGreater(frames_inside, 5)
        # The volume pushed nothing: the box fell through it onto the ground.
        np.testing.assert_allclose(state_0.body_q.numpy()[body][2], _HALF, atol=3.0e-3)


if __name__ == "__main__":
    unittest.main()
