# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Contacts on a cylinder's side store their witness on its axis, like a capsule's."""

import unittest

import numpy as np
import warp as wp

import newton
from newton.tests.unittest_utils import add_function_test, get_test_devices

_RADIUS = 0.08  # [m]
_HALF_HEIGHT = 0.025  # [m]


def _lying_cylinder_contacts(device, spin: float):
    """Collide a cylinder lying on the ground, turned by ``spin`` [rad] about its own axis."""
    builder = newton.ModelBuilder()
    builder.add_ground_plane()
    body = builder.add_body(xform=wp.transform(wp.vec3(0.0, 0.0, _RADIUS - 0.001), wp.quat_identity()))
    # Axis along y.
    builder.add_shape_cylinder(
        body,
        xform=wp.transform(wp.vec3(), wp.quat_from_axis_angle(wp.vec3(1.0, 0.0, 0.0), 0.5 * np.pi)),
        radius=_RADIUS,
        half_height=_HALF_HEIGHT,
    )
    model = builder.finalize(device=device)
    state = model.state()
    pose = state.body_q.numpy()
    pose[body, 3:] = np.asarray(wp.quat_from_axis_angle(wp.vec3(0.0, 1.0, 0.0), spin))
    state.body_q.assign(pose)
    pipeline = newton.CollisionPipeline(model)
    contacts = pipeline.contacts()
    pipeline.collide(state, contacts)
    count = int(contacts.rigid_contact_count.numpy()[0])
    shape_body = model.shape_body.numpy()
    shapes = (contacts.rigid_contact_shape0.numpy()[:count], contacts.rigid_contact_shape1.numpy()[:count])
    side = 1 if shape_body[shapes[1][0]] == body else 0
    points = getattr(contacts, f"rigid_contact_point{side}").numpy()[:count]
    margins = getattr(contacts, f"rigid_contact_margin{side}").numpy()[:count]
    normals = contacts.rigid_contact_normal.numpy()[:count]
    return points, margins, normals, side, pose[body]


def test_side_witness_on_axis(test, device):
    """Side contacts keep their world point but store it on the axis, whichever way the cylinder is turned."""
    for spin in (0.0, 0.7):
        points, margins, normals, side, pose = _lying_cylinder_contacts(device, spin)
        # A contact's normal points from shape 0 to shape 1; the cylinder's outward normal points down.
        down = -normals if side == 1 else normals
        rotation = wp.quat(*pose[3:])
        world = (
            np.array([np.asarray(wp.quat_rotate(rotation, wp.vec3(*point))) for point in points])
            + pose[:3]
            + margins[:, None] * down
        )
        # Side contacts touch the ground below the axis; the manifold may add speculative points elsewhere.
        side_rows = (np.abs(down[:, 2] + 1.0) < 1.0e-3) & (np.abs(world[:, 2]) < 2.0e-3)
        test.assertGreater(int(np.count_nonzero(side_rows)), 0)
        for point, margin in zip(points[side_rows], margins[side_rows], strict=True):
            # Body frame: the axis is local y, through the origin.
            test.assertLess(float(np.hypot(point[0], point[2])), 1.0e-4, f"spin {spin}: witness {point} off the axis")
            test.assertAlmostEqual(float(margin), _RADIUS, delta=1.0e-6)


class TestCylinderSideWitness(unittest.TestCase):
    pass


add_function_test(
    TestCylinderSideWitness, "test_side_witness_on_axis", test_side_witness_on_axis, devices=get_test_devices()
)


if __name__ == "__main__":
    unittest.main(verbosity=2)
