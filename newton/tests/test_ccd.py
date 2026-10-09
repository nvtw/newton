# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Tests for continuous collision detection with ``CollisionPipeline(ccd=True)``."""

import math
import unittest

import numpy as np
import warp as wp

import newton
from newton.tests.unittest_utils import add_function_test, get_cuda_test_devices

FRAME_DT = 1.0 / 60.0
SUBSTEPS = 4


def _builder(gravity=True):
    builder = newton.ModelBuilder() if gravity else newton.ModelBuilder(gravity=wp.vec3(0.0))
    newton.solvers.SolverMuJoCo.register_custom_attributes(builder)
    builder.default_shape_cfg.gap = 0.01
    return builder


def _simulate(model, *, ccd, frames, graph=False):
    """Run SolverMuJoCo with one collide() per frame; return body poses after every frame."""
    pipeline = newton.CollisionPipeline(model, ccd=ccd)
    contacts = pipeline.contacts()
    solver = newton.solvers.SolverMuJoCo(model, use_mujoco_contacts=False, njmax=200, nconmax=100)
    state_0, state_1 = model.state(), model.state()
    control = model.control()
    newton.eval_ik(model, state_0, state_0.joint_q, state_0.joint_qd)

    def frame():
        pipeline.collide(state_0, contacts, dt=FRAME_DT if ccd else None)
        for _ in range(SUBSTEPS):
            state_0.clear_forces()
            solver.step(state_0, state_1, control, contacts, FRAME_DT / SUBSTEPS)
            state_0.assign(state_1)

    if graph:
        with wp.ScopedCapture(device=model.device) as capture:
            frame()
    poses = []
    for _ in range(frames):
        if graph:
            wp.capture_launch(capture.graph)
        else:
            frame()
        poses.append(state_0.body_q.numpy().copy())
    return np.array(poses)


def _bullet_and_wall(device, velocity):
    """A 5 cm box flying at ``velocity`` [m/s] towards a 2 cm thick static wall at x = 0."""
    builder = _builder(gravity=False)
    builder.add_shape_box(-1, hx=0.01, hy=1.0, hz=1.0)
    body = builder.add_body(xform=wp.transform(wp.vec3(-0.5, 0.0, 0.0)))
    builder.add_shape_box(body, hx=0.05, hy=0.05, hz=0.05)
    builder.body_qd[body] = (velocity, 0.0, 0.0, 0.0, 0.0, 0.0)
    return builder.finalize(device=device)


def _add_floor(builder, floor):
    """Add a static 2 m x 2 m floor with its top at z = 0."""
    if floor == "box":
        builder.add_shape_box(-1, xform=wp.transform(wp.vec3(0.0, 0.0, -0.01)), hx=1.0, hy=1.0, hz=0.01)
    elif floor == "mesh":
        vertices = np.array([[-1.0, -1.0, 0.0], [1.0, -1.0, 0.0], [1.0, 1.0, 0.0], [-1.0, 1.0, 0.0]], dtype=np.float32)
        builder.add_shape_mesh(-1, mesh=newton.Mesh(vertices, np.array([0, 1, 2, 0, 2, 3], dtype=np.int32)))
    elif floor == "heightfield":
        heights = np.zeros((9, 9), dtype=np.float32)
        builder.add_shape_heightfield(heightfield=newton.Heightfield(data=heights, nrow=9, ncol=9, hx=1.0, hy=1.0))
    else:
        builder.add_ground_plane()


def test_ccd_prevents_thin_wall_tunneling(test, device):
    model = _bullet_and_wall(device, velocity=120.0)

    without_ccd = _simulate(model, ccd=False, frames=4)
    test.assertGreater(without_ccd[-1, 0, 0], 0.0, "test setup must tunnel without CCD")

    with_ccd = _simulate(model, ccd=True, frames=4)
    # The box front (x + 5 cm) must stop at the wall face at x = -1 cm.
    test.assertLess(with_ccd[:, 0, 0].max() + 0.05, -0.01 + 1.0e-3)


def test_ccd_prevents_floor_tunneling(test, device, floor, speed):
    """A 5 cm sphere falling at ``speed`` [m/s] must land and come to rest on the floor."""
    builder = _builder()
    _add_floor(builder, floor)
    body = builder.add_body(xform=wp.transform(wp.vec3(0.0, 0.0, 1.0)))
    builder.add_shape_sphere(body, radius=0.05)
    builder.body_qd[body] = (0.0, 0.0, -speed, 0.0, 0.0, 0.0)
    model = builder.finalize(device=device)

    without_ccd = _simulate(model, ccd=False, frames=30)
    test.assertLess(without_ccd[-1, 0, 2], 0.0, "test setup must tunnel without CCD")

    with_ccd = _simulate(model, ccd=True, frames=30)
    # The soft contact absorbs the impact: the sphere may dip into the floor but never sinks a full
    # radius, and comes to rest on top.
    test.assertGreater(with_ccd[:, 0, 2].min(), -0.05)
    test.assertAlmostEqual(float(with_ccd[-1, 0, 2]), 0.05, delta=2.0e-3)


def test_ccd_prevents_rotational_tunneling(test, device):
    """A plank spinning a quarter turn per frame must not swing its tip through the wall."""
    builder = _builder(gravity=False)
    builder.add_shape_box(-1, hx=0.01, hy=1.0, hz=1.0)
    body = builder.add_body(xform=wp.transform(wp.vec3(-0.3, 0.0, 0.0)))
    builder.add_shape_box(body, hx=0.02, hy=0.5, hz=0.02)
    builder.body_qd[body] = (0.0, 0.0, 0.0, 0.0, 0.0, -0.5 * math.pi / FRAME_DT)
    model = builder.finalize(device=device)

    def max_tip_x(poses):
        tips = []
        for pose in poses[:, 0]:
            xform = wp.transform(wp.vec3(*pose[:3]), wp.quat(*pose[3:]))
            for y in (-0.5, 0.5):
                tips.append(wp.transform_point(xform, wp.vec3(0.0, y, 0.0))[0])
        return max(tips)

    without_ccd = _simulate(model, ccd=False, frames=2)
    test.assertGreater(max_tip_x(without_ccd), 0.01, "test setup must tunnel without CCD")

    with_ccd = _simulate(model, ccd=True, frames=6)
    test.assertLess(max_tip_x(with_ccd), 0.0)


def test_ccd_stops_articulated_link(test, device):
    """A 1 m link on a hinge, swinging with a 40 m/s tip, must stop at a 2 cm wall."""
    builder = _builder(gravity=False)
    builder.add_shape_box(-1, xform=wp.transform(wp.vec3(0.0, 0.5, 0.0)), hx=1.0, hy=0.01, hz=0.2)
    link = builder.add_link(xform=wp.transform(wp.vec3(0.5, 0.0, 0.0)))
    builder.add_shape_box(link, hx=0.5, hy=0.02, hz=0.02)
    joint = builder.add_joint_revolute(
        -1,
        link,
        parent_xform=wp.transform_identity(),
        child_xform=wp.transform(wp.vec3(-0.5, 0.0, 0.0)),
        axis=(0, 0, 1),
    )
    builder.add_articulation([joint])
    # 40 rad/s about the hinge at the origin; the link COM sits 0.5 m out along x.
    builder.body_qd[link] = (0.0, 20.0, 0.0, 0.0, 0.0, 40.0)
    model = builder.finalize(device=device)

    def max_tip_y(poses):
        return max(
            wp.transform_point(wp.transform(wp.vec3(*p[:3]), wp.quat(*p[3:])), wp.vec3(0.5, 0.02, 0.0))[1]
            for p in poses[:, 0]
        )

    without_ccd = _simulate(model, ccd=False, frames=3)
    test.assertGreater(max_tip_y(without_ccd), 0.51, "test setup must tunnel without CCD")

    with_ccd = _simulate(model, ccd=True, frames=6)
    # The wall face is at y = 0.49.
    test.assertLess(max_tip_y(with_ccd), 0.49 + 1.0e-3)


def test_ccd_stops_moving_pair(test, device):
    """Two 5 cm boxes flying at each other at 60 m/s each must not pass through each other."""
    builder = _builder(gravity=False)
    bodies = []
    for x, v in ((-0.5, 60.0), (0.5, -60.0)):
        body = builder.add_body(xform=wp.transform(wp.vec3(x, 0.0, 0.0)))
        builder.add_shape_box(body, hx=0.05, hy=0.05, hz=0.05)
        builder.body_qd[body] = (v, 0.0, 0.0, 0.0, 0.0, 0.0)
        bodies.append(body)
    model = builder.finalize(device=device)

    without_ccd = _simulate(model, ccd=False, frames=4)
    test.assertGreater(without_ccd[-1, 0, 0], without_ccd[-1, 1, 0], "test setup must tunnel without CCD")

    with_ccd = _simulate(model, ccd=True, frames=6)
    gaps = with_ccd[:, 1, 0] - with_ccd[:, 0, 0]
    test.assertGreater(gaps.min(), 0.1 - 1.0e-3)


def test_ccd_keeps_fast_sliding_contact(test, device):
    """A box sliding fast on the ground must slide exactly as without CCD."""
    builder = _builder()
    builder.default_shape_cfg.mu = 0.5
    _add_floor(builder, "plane")
    body = builder.add_body(xform=wp.transform(wp.vec3(-0.9, 0.0, 0.1)))
    builder.add_shape_box(body, hx=0.1, hy=0.1, hz=0.1)
    builder.body_qd[body] = (30.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    model = builder.finalize(device=device)

    with_ccd = _simulate(model, ccd=True, frames=3)
    without_ccd = _simulate(model, ccd=False, frames=3)
    test.assertGreater(with_ccd[-1, 0, 0], 0.0)
    np.testing.assert_allclose(with_ccd, without_ccd, atol=1.0e-4)


def test_ccd_graph_capture(test, device):
    model = _bullet_and_wall(device, velocity=120.0)
    eager = _simulate(model, ccd=True, frames=4)
    captured = _simulate(model, ccd=True, frames=4, graph=True)
    np.testing.assert_allclose(captured, eager, atol=1.0e-5)


class TestCCD(unittest.TestCase):
    pass


devices = get_cuda_test_devices()
add_function_test(
    TestCCD, "test_ccd_prevents_thin_wall_tunneling", test_ccd_prevents_thin_wall_tunneling, devices=devices
)
# A 2 cm box floor at a fall speed MuJoCo's default soft contact can absorb; one-sided surfaces faster.
for _floor, _speed in (("box", 10.0), ("mesh", 30.0), ("heightfield", 30.0)):
    add_function_test(
        TestCCD,
        f"test_ccd_prevents_floor_tunneling_{_floor}",
        test_ccd_prevents_floor_tunneling,
        devices=devices,
        floor=_floor,
        speed=_speed,
    )
add_function_test(
    TestCCD, "test_ccd_prevents_rotational_tunneling", test_ccd_prevents_rotational_tunneling, devices=devices
)
add_function_test(TestCCD, "test_ccd_stops_articulated_link", test_ccd_stops_articulated_link, devices=devices)
add_function_test(TestCCD, "test_ccd_stops_moving_pair", test_ccd_stops_moving_pair, devices=devices)
add_function_test(TestCCD, "test_ccd_keeps_fast_sliding_contact", test_ccd_keeps_fast_sliding_contact, devices=devices)
add_function_test(TestCCD, "test_ccd_graph_capture", test_ccd_graph_capture, devices=devices)


if __name__ == "__main__":
    unittest.main(verbosity=2)
