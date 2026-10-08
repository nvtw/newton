# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Tests for opt-in continuous collision detection of fast rigid bodies."""

import math
import unittest

import numpy as np
import warp as wp

import newton
from newton.tests.unittest_utils import add_function_test, get_cuda_test_devices, get_test_devices

DT = 1.0 / 60.0


def _simulate(model, solver, *, ccd, steps, graph=False):
    """Step ``model`` and return the body poses after every step, shape [steps, body_count, 7]."""
    pipeline = newton.CollisionPipeline(model, broad_phase="nxn", ccd=ccd)
    contacts = pipeline.contacts()
    state_0, state_1 = model.state(), model.state()
    newton.eval_ik(model, state_0, state_0.joint_q, state_0.joint_qd)

    def step():
        pipeline.collide(state_0, contacts, dt=DT)
        solver.step(state_0, state_1, None, contacts, DT)
        pipeline.resolve_ccd(state_1)
        wp.copy(state_0.body_q, state_1.body_q)
        wp.copy(state_0.body_qd, state_1.body_qd)
        wp.copy(state_0.joint_q, state_1.joint_q)
        wp.copy(state_0.joint_qd, state_1.joint_qd)

    if graph:
        with wp.ScopedCapture(device=model.device) as capture:
            step()

    poses = []
    for _ in range(steps):
        if graph:
            wp.capture_launch(capture.graph)
        else:
            step()
        poses.append(state_0.body_q.numpy().copy())
    return np.array(poses)


def _bullet_and_wall_model(device, velocity):
    """Build a small box flying at ``velocity`` [m/s] towards a 2 cm thick static wall at x = 0."""
    builder = newton.ModelBuilder(gravity=wp.vec3(0.0))
    builder.add_shape_box(-1, hx=0.01, hy=1.0, hz=1.0)
    body = builder.add_body(xform=wp.transform(wp.vec3(-0.5, 0.0, 0.0)))
    builder.add_shape_box(body, hx=0.05, hy=0.05, hz=0.05)
    builder.body_qd[body] = (velocity, 0.0, 0.0, 0.0, 0.0, 0.0)
    return builder.finalize(device=device)


def _mesh_floor():
    vertices = np.array([[-1.0, -1.0, 0.0], [1.0, -1.0, 0.0], [1.0, 1.0, 0.0], [-1.0, 1.0, 0.0]], dtype=np.float32)
    return newton.Mesh(vertices, np.array([0, 1, 2, 0, 2, 3], dtype=np.int32))


def _make_solver(name, model):
    if name == "xpbd":
        return newton.solvers.SolverXPBD(model, iterations=4)
    if name == "semi_implicit":
        return newton.solvers.SolverSemiImplicit(model)
    if name == "featherstone":
        return newton.solvers.SolverFeatherstone(model)
    raise ValueError(name)


def test_ccd_prevents_thin_wall_tunneling(test, device, solver_name):
    model = _bullet_and_wall_model(device, velocity=120.0)
    solver = _make_solver(solver_name, model)

    without_ccd = _simulate(model, solver, ccd=False, steps=4)
    test.assertGreater(without_ccd[-1, 0, 0], 0.0, "test setup must tunnel without CCD")

    with_ccd = _simulate(model, solver, ccd=True, steps=4)
    # The box (half-extent 5 cm) must stay left of the wall face at x = -1 cm.
    test.assertLess(with_ccd[:, 0, 0].max(), -0.01 - 0.05 + 0.01)


def test_ccd_prevents_floor_tunneling(test, device):
    """A sphere falling at 200 m/s must land on a 2 cm thick static floor."""
    builder = newton.ModelBuilder()
    builder.add_shape_box(-1, hx=1.0, hy=1.0, hz=0.01)
    body = builder.add_body(xform=wp.transform(wp.vec3(0.0, 0.0, 1.0)))
    builder.add_shape_sphere(body, radius=0.05)
    builder.body_qd[body] = (0.0, 0.0, -200.0, 0.0, 0.0, 0.0)
    model = builder.finalize(device=device)
    solver = newton.solvers.SolverXPBD(model, iterations=4)

    without_ccd = _simulate(model, solver, ccd=False, steps=3)
    test.assertLess(without_ccd[-1, 0, 2], -0.01, "test setup must tunnel without CCD")

    with_ccd = _simulate(model, solver, ccd=True, steps=10)
    test.assertGreater(with_ccd[:, 0, 2].min(), 0.03)


def test_ccd_prevents_mesh_tunneling(test, device):
    """A sphere falling at 200 m/s must land on a zero-thickness triangle mesh floor."""
    builder = newton.ModelBuilder()
    builder.add_shape_mesh(-1, mesh=_mesh_floor())
    body = builder.add_body(xform=wp.transform(wp.vec3(0.0, 0.0, 1.0)))
    builder.add_shape_sphere(body, radius=0.05)
    builder.body_qd[body] = (0.0, 0.0, -200.0, 0.0, 0.0, 0.0)
    model = builder.finalize(device=device)
    solver = newton.solvers.SolverXPBD(model, iterations=4)

    without_ccd = _simulate(model, solver, ccd=False, steps=3)
    test.assertLess(without_ccd[-1, 0, 2], 0.0, "test setup must tunnel without CCD")

    with_ccd = _simulate(model, solver, ccd=True, steps=10)
    test.assertGreater(with_ccd[:, 0, 2].min(), 0.03)


def test_ccd_prevents_rotational_tunneling(test, device):
    """A plank spinning a quarter turn per step must not swing its tip through the wall."""
    builder = newton.ModelBuilder(gravity=wp.vec3(0.0))
    builder.add_shape_box(-1, hx=0.01, hy=1.0, hz=1.0)
    body = builder.add_body(xform=wp.transform(wp.vec3(-0.3, 0.0, 0.0)))
    builder.add_shape_box(body, hx=0.02, hy=0.5, hz=0.02)
    builder.body_qd[body] = (0.0, 0.0, 0.0, 0.0, 0.0, -0.5 * math.pi / DT)
    model = builder.finalize(device=device)
    solver = newton.solvers.SolverXPBD(model, iterations=4)

    def max_tip_x(poses):
        tips = []
        for pose in poses[:, 0]:
            xform = wp.transform(wp.vec3(*pose[:3]), wp.quat(*pose[3:]))
            for y in (-0.5, 0.5):
                tips.append(wp.transform_point(xform, wp.vec3(0.0, y, 0.0))[0])
        return max(tips)

    without_ccd = _simulate(model, solver, ccd=False, steps=1)
    test.assertGreater(max_tip_x(without_ccd), 0.01, "test setup must tunnel without CCD")

    with_ccd = _simulate(model, solver, ccd=True, steps=6)
    test.assertLess(max_tip_x(with_ccd), 0.01)


def test_ccd_keeps_fast_sliding_contact(test, device):
    """A box sliding fast on the ground is a fast body but must not be stopped by its resting contact."""
    for floor in ("plane", "mesh"):
        builder = newton.ModelBuilder()
        builder.default_shape_cfg.mu = 0.0
        if floor == "plane":
            builder.add_ground_plane()
        else:
            builder.add_shape_mesh(-1, mesh=_mesh_floor())
        body = builder.add_body(xform=wp.transform(wp.vec3(-0.9, 0.0, 0.1)))
        builder.add_shape_box(body, hx=0.1, hy=0.1, hz=0.1)
        builder.body_qd[body] = (30.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        model = builder.finalize(device=device)
        solver = newton.solvers.SolverXPBD(model, iterations=4)

        poses = _simulate(model, solver, ccd=True, steps=3)
        np.testing.assert_allclose(poses[:, 0, 0] + 0.9, 30.0 * DT * np.arange(1, 4), rtol=0.05, err_msg=floor)


def test_ccd_graph_capture(test, device):
    model = _bullet_and_wall_model(device, velocity=120.0)
    solver = newton.solvers.SolverXPBD(model, iterations=4)
    eager = _simulate(model, solver, ccd=True, steps=4)
    captured = _simulate(model, solver, ccd=True, steps=4, graph=True)
    np.testing.assert_allclose(captured, eager, atol=1.0e-6)


class TestCCD(unittest.TestCase):
    pass


for _solver_name in ("xpbd", "semi_implicit", "featherstone"):
    add_function_test(
        TestCCD,
        f"test_ccd_prevents_thin_wall_tunneling_{_solver_name}",
        test_ccd_prevents_thin_wall_tunneling,
        devices=get_test_devices(),
        solver_name=_solver_name,
    )
add_function_test(
    TestCCD, "test_ccd_prevents_floor_tunneling", test_ccd_prevents_floor_tunneling, devices=get_test_devices()
)
add_function_test(
    TestCCD, "test_ccd_prevents_mesh_tunneling", test_ccd_prevents_mesh_tunneling, devices=get_test_devices()
)
add_function_test(
    TestCCD,
    "test_ccd_prevents_rotational_tunneling",
    test_ccd_prevents_rotational_tunneling,
    devices=get_test_devices(),
)
add_function_test(
    TestCCD, "test_ccd_keeps_fast_sliding_contact", test_ccd_keeps_fast_sliding_contact, devices=get_test_devices()
)
add_function_test(TestCCD, "test_ccd_graph_capture", test_ccd_graph_capture, devices=get_cuda_test_devices())


if __name__ == "__main__":
    unittest.main(verbosity=2)
