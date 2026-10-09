# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Tests for continuous collision detection with ``CollisionPipeline(ccd=True)``."""

import math
import unittest
import warnings

import numpy as np
import warp as wp

import newton
from newton.tests.unittest_utils import add_function_test, get_cuda_test_devices

FRAME_DT = 1.0 / 60.0
SUBSTEPS = 4


def _builder(gravity=True):
    builder = newton.ModelBuilder() if gravity else newton.ModelBuilder(gravity=wp.vec3(0.0))
    newton.solvers.SolverMuJoCo.register_custom_attributes(builder)
    newton.solvers.SolverKamino.register_custom_attributes(builder)
    builder.default_shape_cfg.gap = 0.01
    return builder


def _make_solver(name, model):
    if name == "kamino":
        return newton.solvers.SolverKamino(model, config=newton.solvers.SolverKamino.Config.from_model(model))
    return newton.solvers.SolverMuJoCo(model, use_mujoco_contacts=False, njmax=200, nconmax=100)


def _simulate(model, *, ccd, frames, solver="mujoco", graph=False, substeps=SUBSTEPS):
    """Run ``solver`` with one collide() per frame; return body poses after every frame."""
    pipeline = newton.CollisionPipeline(model, ccd=ccd)
    contacts = pipeline.contacts()
    solver = _make_solver(solver, model)
    state_0, state_1 = model.state(), model.state()
    control = model.control()
    newton.eval_ik(model, state_0, state_0.joint_q, state_0.joint_qd)

    def frame():
        pipeline.collide(state_0, contacts, dt=FRAME_DT if ccd else None)
        for _ in range(substeps):
            state_0.clear_forces()
            solver.step(state_0, state_1, control, contacts, FRAME_DT / substeps)
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


def test_ccd_prevents_thin_wall_tunneling(test, device, solver_name, substeps):
    """Stop a box at 120 m/s in a 2 cm wall instead of letting it pass through."""
    model = _bullet_and_wall(device, velocity=120.0)

    without_ccd = _simulate(model, solver=solver_name, ccd=False, frames=4, substeps=substeps)
    test.assertGreater(without_ccd[-1, 0, 0], 0.0, "test setup must tunnel without CCD")

    with_ccd = _simulate(model, solver=solver_name, ccd=True, frames=4, substeps=substeps)
    # The box front (x + 5 cm) may enter the wall (face at x = -1 cm) by at most a quarter of the
    # wall's 2 cm thickness.
    test.assertLess(with_ccd[:, 0, 0].max() + 0.05, -0.01 + 0.005 + 1.0e-3)


def test_ccd_prevents_floor_tunneling(test, device, floor, speed, solver_name):
    """Land a 5 cm sphere falling at ``speed`` [m/s] on the floor and bring it to rest there."""
    builder = _builder()
    _add_floor(builder, floor)
    body = builder.add_body(xform=wp.transform(wp.vec3(0.0, 0.0, 1.0)))
    builder.add_shape_sphere(body, radius=0.05)
    builder.body_qd[body] = (0.0, 0.0, -speed, 0.0, 0.0, 0.0)
    model = builder.finalize(device=device)

    without_ccd = _simulate(model, solver=solver_name, ccd=False, frames=30)
    test.assertLess(without_ccd[:, 0, 2].min(), 0.0, "test setup must sink below the floor without CCD")

    with_ccd = _simulate(model, solver=solver_name, ccd=True, frames=30)
    # The soft contact absorbs the impact: the sphere may dip into the floor but never sinks a full
    # radius, and comes to rest on top.
    test.assertGreater(with_ccd[:, 0, 2].min(), -0.05)
    test.assertAlmostEqual(float(with_ccd[-1, 0, 2]), 0.05, delta=2.0e-3)


def test_ccd_prevents_rotational_tunneling(test, device, solver_name):
    """Keep a plank spinning a quarter turn per frame from swinging its tip through a wall."""
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

    without_ccd = _simulate(model, solver=solver_name, ccd=False, frames=2)
    test.assertGreater(max_tip_x(without_ccd), 0.01, "test setup must tunnel without CCD")

    with_ccd = _simulate(model, solver=solver_name, ccd=True, frames=6)
    test.assertLess(max_tip_x(with_ccd), 0.0)


def test_ccd_stops_articulated_link(test, device, solver_name):
    """Stop a 1 m link on a hinge, swinging with a 40 m/s tip, at a 2 cm wall."""
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

    without_ccd = _simulate(model, solver=solver_name, ccd=False, frames=3)
    test.assertGreater(max_tip_y(without_ccd), 0.51, "test setup must tunnel without CCD")

    with_ccd = _simulate(model, solver=solver_name, ccd=True, frames=6)
    # The wall face is at y = 0.49.
    test.assertLess(max_tip_y(with_ccd), 0.49 + 1.0e-3)


def test_ccd_stops_moving_pair(test, device, solver_name):
    """Keep two 5 cm boxes flying at each other at 60 m/s each from passing through each other."""
    builder = _builder(gravity=False)
    for x, v in ((-0.5, 60.0), (0.5, -60.0)):
        body = builder.add_body(xform=wp.transform(wp.vec3(x, 0.0, 0.0)))
        builder.add_shape_box(body, hx=0.05, hy=0.05, hz=0.05)
        builder.body_qd[body] = (v, 0.0, 0.0, 0.0, 0.0, 0.0)
    model = builder.finalize(device=device)

    without_ccd = _simulate(model, solver=solver_name, ccd=False, frames=4)
    test.assertGreater(without_ccd[-1, 0, 0], without_ccd[-1, 1, 0], "test setup must tunnel without CCD")

    with_ccd = _simulate(model, solver=solver_name, ccd=True, frames=6)
    # The boxes (10 cm wide) may overlap by at most a quarter of their width.
    gaps = with_ccd[:, 1, 0] - with_ccd[:, 0, 0]
    test.assertGreater(gaps.min(), 0.1 - 0.025 - 1.0e-3)


def test_ccd_bounds_mesh_link_on_surface(test, device, solver_name):
    """Bound how deep a body with a triangle-mesh collision shape, as imported robot links have,
    sinks into a triangle-mesh floor when it lands at 30 m/s."""
    builder = _builder()
    _add_floor(builder, "mesh")
    body = builder.add_body(xform=wp.transform(wp.vec3(0.0, 0.0, 1.0)))
    builder.add_shape_mesh(body, mesh=newton.Mesh.create_box(0.05, 0.05, 0.05))
    builder.body_qd[body] = (0.0, 0.0, -30.0, 0.0, 0.0, 0.0)
    model = builder.finalize(device=device)

    with_ccd = _simulate(model, solver=solver_name, ccd=True, frames=30)
    # The box bottom (z - 5 cm) may sink by at most half its 5 cm half-thickness, and the box comes
    # to rest on top.
    test.assertGreater(with_ccd[:, 0, 2].min() - 0.05, -0.025 - 1.0e-3)
    test.assertAlmostEqual(float(with_ccd[-1, 0, 2]), 0.05, delta=2.0e-3)


def test_ccd_stops_bullets_in_each_world(test, device, solver_name):
    """Stop a box in a 2 cm wall in every world, with a different speed per world."""
    template = _builder(gravity=False)
    template.add_shape_box(-1, hx=0.01, hy=1.0, hz=1.0)
    body = template.add_body(xform=wp.transform(wp.vec3(-0.5, 0.0, 0.0)))
    template.add_shape_box(body, hx=0.05, hy=0.05, hz=0.05)
    builder = _builder(gravity=False)
    speeds = (20.0, 60.0, 120.0)
    builder.replicate(template, len(speeds), spacing=(0.0, 3.0, 0.0))
    for world, speed in enumerate(speeds):
        builder.body_qd[world] = (speed, 0.0, 0.0, 0.0, 0.0, 0.0)
    model = builder.finalize(device=device)

    with_ccd = _simulate(model, solver=solver_name, ccd=True, frames=4)
    # Each box front may enter its world's wall by at most a quarter of the wall's thickness.
    test.assertLess(with_ccd[:, :, 0].max() + 0.05, -0.01 + 0.005 + 1.0e-3)


def test_ccd_ignores_near_miss(test, device, solver_name, wall):
    """Keep the velocity of a box passing 1 cm beside a wall edge at 5 m/s: contacts whose shapes
    never touch must not act as ghost walls."""
    builder = _builder(gravity=False)
    if wall == "mesh":
        builder.add_shape_mesh(-1, mesh=newton.Mesh.create_box(0.01, 0.5, 0.5, compute_inertia=False))
    else:
        builder.add_shape_box(-1, hx=0.01, hy=0.5, hz=0.5)
    body = builder.add_body(xform=wp.transform(wp.vec3(-0.5, 0.56, 0.0)))
    builder.add_shape_box(body, hx=0.05, hy=0.05, hz=0.05)
    builder.body_qd[body] = (5.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    model = builder.finalize(device=device)

    poses = _simulate(model, solver=solver_name, ccd=True, frames=12)
    # Free flight: x advances 5 m/s * frame_dt per frame, y and z stay put.
    expected_x = -0.5 + 5.0 * FRAME_DT * np.arange(1, 13)
    np.testing.assert_allclose(poses[:, 0, 0], expected_x, atol=1.0e-4)
    np.testing.assert_allclose(poses[:, 0, 1:3], np.tile([0.56, 0.0], (12, 1)), atol=1.0e-4)


def test_ccd_keeps_contact_compliance(test, device, solver_name):
    """Land a slowly falling sphere with the contact's authored compliance, as without CCD when the
    contact gap already detects the impact in time: CCD must not stiffen ordinary impacts such as
    footfalls."""
    builder = _builder()
    builder.default_shape_cfg.ke = 2.0e4
    builder.default_shape_cfg.kd = 2.0e2
    # Wider than one frame of motion, so the run without CCD detects the landing in time too.
    builder.default_shape_cfg.gap = 0.05
    _add_floor(builder, "plane")
    body = builder.add_body(xform=wp.transform(wp.vec3(0.0, 0.0, 0.1)))
    builder.add_shape_sphere(body, radius=0.05)
    builder.body_qd[body] = (0.0, 0.0, -2.0, 0.0, 0.0, 0.0)
    model = builder.finalize(device=device)

    with_ccd = _simulate(model, solver=solver_name, ccd=True, frames=20)
    without_ccd = _simulate(model, solver=solver_name, ccd=False, frames=20)
    # The soft landing sinks the sphere a few millimetres; both runs must agree on it.
    test.assertLess(without_ccd[:, 0, 2].min(), 0.05 - 1.0e-3, "test setup must penetrate softly")
    np.testing.assert_allclose(with_ccd[:, 0, 2], without_ccd[:, 0, 2], atol=1.0e-3)


def test_ccd_keeps_fast_sliding_contact(test, device, solver_name):
    """Slide a fast box along the ground exactly as without CCD."""
    builder = _builder()
    builder.default_shape_cfg.mu = 0.5
    _add_floor(builder, "plane")
    body = builder.add_body(xform=wp.transform(wp.vec3(-0.9, 0.0, 0.1)))
    builder.add_shape_box(body, hx=0.1, hy=0.1, hz=0.1)
    builder.body_qd[body] = (30.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    model = builder.finalize(device=device)

    with_ccd = _simulate(model, solver=solver_name, ccd=True, frames=3)
    without_ccd = _simulate(model, solver=solver_name, ccd=False, frames=3)
    test.assertGreater(with_ccd[-1, 0, 0], 0.0)
    np.testing.assert_allclose(with_ccd, without_ccd, atol=1.0e-4)


def test_ccd_graph_capture(test, device, solver_name):
    """Produce the same impact from a captured CUDA graph as from eager launches."""
    model = _bullet_and_wall(device, velocity=120.0)
    eager = _simulate(model, solver=solver_name, ccd=True, frames=4)
    captured = _simulate(model, solver=solver_name, ccd=True, frames=4, graph=True)
    np.testing.assert_allclose(captured, eager, atol=1.0e-5)


def test_ccd_warns_when_solver_does_not_enforce(test, device):
    """Warn once from solvers that treat ccd=True contacts like regular contacts, never from enforcing ones."""
    model = _bullet_and_wall(device, velocity=5.0)
    for solver, ccd, expected in (
        (newton.solvers.SolverXPBD(model), True, 1),
        (newton.solvers.SolverXPBD(model), False, 0),
        (_make_solver("mujoco", model), True, 0),
    ):
        pipeline = newton.CollisionPipeline(model, ccd=ccd, speculative_contact_gap_max=None if ccd else 0.5)
        contacts = pipeline.contacts()
        state_0, state_1 = model.state(), model.state()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            for _ in range(2):
                pipeline.collide(state_0, contacts, dt=FRAME_DT)
                solver.step(state_0, state_1, model.control(), contacts, FRAME_DT)
        messages = [w for w in caught if "CollisionPipeline(ccd=True)" in str(w.message)]
        test.assertEqual(len(messages), expected, type(solver).__name__)


class TestCCD(unittest.TestCase):
    pass


devices = get_cuda_test_devices()
for _solver in ("mujoco", "kamino"):
    # One substep per frame catches impacts resolved before the bodies touch; four is typical.
    for _substeps in (1, 4):
        add_function_test(
            TestCCD,
            f"test_ccd_prevents_thin_wall_tunneling_{_substeps}_substeps_{_solver}",
            test_ccd_prevents_thin_wall_tunneling,
            devices=devices,
            solver_name=_solver,
            substeps=_substeps,
        )
    # A 2 cm box floor at a fall speed the default soft contact can absorb; one-sided surfaces faster.
    for _floor, _speed in (("box", 10.0), ("mesh", 30.0), ("heightfield", 30.0)):
        add_function_test(
            TestCCD,
            f"test_ccd_prevents_floor_tunneling_{_floor}_{_solver}",
            test_ccd_prevents_floor_tunneling,
            devices=devices,
            floor=_floor,
            speed=_speed,
            solver_name=_solver,
        )
    for _name, _func in (
        ("test_ccd_prevents_rotational_tunneling", test_ccd_prevents_rotational_tunneling),
        ("test_ccd_stops_articulated_link", test_ccd_stops_articulated_link),
        ("test_ccd_stops_moving_pair", test_ccd_stops_moving_pair),
        ("test_ccd_bounds_mesh_link_on_surface", test_ccd_bounds_mesh_link_on_surface),
        ("test_ccd_stops_bullets_in_each_world", test_ccd_stops_bullets_in_each_world),
        ("test_ccd_keeps_fast_sliding_contact", test_ccd_keeps_fast_sliding_contact),
        ("test_ccd_graph_capture", test_ccd_graph_capture),
    ):
        add_function_test(TestCCD, f"{_name}_{_solver}", _func, devices=devices, solver_name=_solver)
    for _wall in ("box", "mesh"):
        add_function_test(
            TestCCD,
            f"test_ccd_ignores_near_miss_{_wall}_{_solver}",
            test_ccd_ignores_near_miss,
            devices=devices,
            solver_name=_solver,
            wall=_wall,
        )
add_function_test(
    TestCCD,
    "test_ccd_warns_when_solver_does_not_enforce",
    test_ccd_warns_when_solver_does_not_enforce,
    devices=devices,
)
# Kamino enforces speculative contacts in its velocity-level solve; compliance is a MuJoCo property.
add_function_test(
    TestCCD,
    "test_ccd_keeps_contact_compliance_mujoco",
    test_ccd_keeps_contact_compliance,
    devices=devices,
    solver_name="mujoco",
)


if __name__ == "__main__":
    unittest.main(verbosity=2)
