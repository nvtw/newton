# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Penetration-free rigid-rigid collisions with :class:`~newton.solvers.SolverPhoenX`.

Each scenario runs through a captured CUDA graph and checks penetration with
an independent analytic measurement of every body's lowest point, so a
failure of the pipeline's own distance queries cannot hide a violation.
"""

from __future__ import annotations

import math
import unittest

import numpy as np
import warp as wp

import newton
from newton._src.geometry.penetration_free import (
    STATUS_CONSTRAINT_OVERFLOW,
    STATUS_FATAL_MASK,
    STATUS_KINEMATIC_VIOLATION,
    STATUS_REFERENCE_MISMATCH,
)
from newton.geometry import PenetrationFreeConfig
from newton.solvers import SolverPhoenX

_DT = 1.0 / 60.0


def require_cuda_graph_capture(label: str):
    device = wp.get_preferred_device()
    if not device.is_cuda or not wp.is_mempool_enabled(device):
        raise unittest.SkipTest(f"{label} require CUDA graph capture with Warp mempool enabled.")
    return device


_TOLERANCE = 1.0e-3
# Float32 slack on top of the configured tolerance for the analytic checks [m].
_MEASURE_SLACK = 2.0e-4


def _box_mesh_data(hx: float, hy: float, hz: float) -> tuple[np.ndarray, np.ndarray]:
    vertices = np.array(
        [[x, y, z] for x in (-hx, hx) for y in (-hy, hy) for z in (-hz, hz)],
        dtype=np.float32,
    )
    faces = np.array(
        [
            [0, 1, 3], [0, 3, 2], [4, 6, 7], [4, 7, 5], [0, 4, 5], [0, 5, 1],
            [2, 3, 7], [2, 7, 6], [0, 2, 6], [0, 6, 4], [1, 5, 7], [1, 7, 3],
        ],
        dtype=np.int32,
    )  # fmt: skip
    return vertices, faces.flatten()


def _box_mesh(hx: float, hy: float, hz: float) -> newton.Mesh:
    vertices, indices = _box_mesh_data(hx, hy, hz)
    return newton.Mesh(vertices, indices)


def _rotation_matrix(q) -> np.ndarray:
    x, y, z, w = (float(v) for v in q)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def _lowest_point(kind: str, params: tuple, body_q: np.ndarray) -> float:
    """Return the exact lowest world z of a shape centered at its body origin."""
    p = body_q[:3]
    r = _rotation_matrix(body_q[3:])
    down = r.T @ np.array([0.0, 0.0, -1.0])  # world -z in the shape frame
    if kind == "sphere":
        return p[2] - params[0]
    if kind == "capsule":
        radius, half_height = params
        return p[2] - radius - half_height * abs(down[2])
    if kind == "ellipsoid":
        return p[2] - math.sqrt(sum((params[i] * down[i]) ** 2 for i in range(3)))
    if kind == "cylinder":
        radius, half_height = params
        lateral = math.hypot(down[0], down[1])
        return p[2] - radius * lateral - half_height * abs(down[2])
    if kind == "cone":
        radius, half_height = params
        lateral = math.hypot(down[0], down[1])
        apex = half_height * down[2]
        base = radius * lateral - half_height * down[2]
        return p[2] - max(apex, base)
    # Polyhedra: params holds the local vertices.
    vertices = params[0]
    return float(np.min((vertices @ r.T)[:, 2])) + p[2]


_SHAPES = ("sphere", "box", "capsule", "cylinder", "cone", "ellipsoid", "convex", "mesh")


def _add_shape(builder: newton.ModelBuilder, body: int, kind: str):
    if kind == "sphere":
        builder.add_shape_sphere(body, radius=0.2)
        return ("sphere", (0.2,))
    if kind == "box":
        builder.add_shape_box(body, hx=0.2, hy=0.15, hz=0.1)
        return ("poly", (_box_mesh_data(0.2, 0.15, 0.1)[0],))
    if kind == "capsule":
        builder.add_shape_capsule(body, radius=0.1, half_height=0.2)
        return ("capsule", (0.1, 0.2))
    if kind == "cylinder":
        builder.add_shape_cylinder(body, radius=0.15, half_height=0.2)
        return ("cylinder", (0.15, 0.2))
    if kind == "cone":
        builder.add_shape_cone(body, radius=0.2, half_height=0.2)
        return ("cone", (0.2, 0.2))
    if kind == "ellipsoid":
        builder.add_shape_ellipsoid(body, rx=0.2, ry=0.1, rz=0.15)
        return ("ellipsoid", (0.2, 0.1, 0.15))
    if kind == "convex":
        builder.add_shape_convex_hull(body, mesh=_box_mesh(0.2, 0.1, 0.15))
        return ("poly", (_box_mesh_data(0.2, 0.1, 0.15)[0],))
    builder.add_shape_mesh(body, mesh=_box_mesh(0.2, 0.1, 0.15))
    return ("poly", (_box_mesh_data(0.2, 0.1, 0.15)[0],))


def _add_ground(builder: newton.ModelBuilder, ground: str) -> None:
    """Add a static collider whose top surface is z = 0."""
    if ground == "plane":
        builder.add_ground_plane()
    elif ground == "box":
        builder.add_shape_box(-1, xform=wp.transform((0.0, 0.0, -0.05), wp.quat_identity()), hx=20, hy=20, hz=0.05)
    elif ground == "mesh":
        builder.add_shape_mesh(
            -1, xform=wp.transform((0.0, 0.0, -0.05), wp.quat_identity()), mesh=_box_mesh(20.0, 20.0, 0.05)
        )
    elif ground == "heightfield":
        heightfield = newton.Heightfield(data=np.zeros((21, 21), dtype=np.float32), nrow=21, ncol=21, hx=20.0, hy=20.0)
        builder.add_shape_heightfield(xform=wp.transform((0.0, 0.0, 0.0), wp.quat_identity()), heightfield=heightfield)
    else:
        raise ValueError(ground)


class _CapturedRun:
    """Collide-and-step loop replayed from one captured CUDA graph per frame."""

    def __init__(self, model, pipeline, solver, state):
        self.device = require_cuda_graph_capture("Penetration-free PhoenX tests")
        self.model = model
        self.pipeline = pipeline
        self.solver = solver
        self.contacts = pipeline.contacts()
        self.state_0 = state
        self.state_1 = model.state()
        self.graph = None

    def _frame(self):
        self.pipeline.collide(self.state_0, self.contacts, dt=_DT)
        self.solver.step(self.state_0, self.state_1, None, self.contacts, _DT)
        wp.copy(self.state_0.body_q, self.state_1.body_q)
        wp.copy(self.state_0.body_qd, self.state_1.body_qd)
        self.state_0.clear_forces()

    def step(self):
        if self.graph is None:
            self._frame()
            with wp.ScopedCapture(device=self.device) as capture:
                self._frame()
            self.graph = capture.graph
            return
        wp.capture_launch(self.graph)

    @property
    def data(self):
        return self.contacts._penetration_free

    def failure_bits(self) -> int:
        """Return raised status bits that void the guarantee; budget exhaustion only stalls motion."""
        return int(self.data.status.numpy()[0]) & (int(STATUS_FATAL_MASK) | int(STATUS_KINEMATIC_VIOLATION))


def _make_run(builder, *, penetration_free, broad_phase="sap"):
    model = builder.finalize(device=require_cuda_graph_capture("Penetration-free PhoenX tests"))
    pipeline = newton.CollisionPipeline(
        model,
        broad_phase=broad_phase,
        contact_matching="sticky",
        penetration_free=penetration_free,
    )
    solver = SolverPhoenX(model, collision_pipeline=pipeline)
    return _CapturedRun(model, pipeline, solver, model.state())


class TestPenetrationFreePhoenX(unittest.TestCase):
    def _head_on_spheres(self, penetration_free):
        builder = newton.ModelBuilder(gravity=wp.vec3(0.0))
        for x in (-1.0, 1.0):
            body = builder.add_body(xform=wp.transform((x, 0.0, 0.0), wp.quat_identity()))
            builder.add_shape_sphere(body, radius=0.1)
        run = _make_run(builder, penetration_free=penetration_free, broad_phase="nxn")
        qd = run.state_0.body_qd.numpy()
        qd[0, 0], qd[1, 0] = 120.0, -120.0
        run.state_0.body_qd.assign(qd)
        min_gap = math.inf
        crossed = False
        for _ in range(20):
            run.step()
            x = run.state_0.body_q.numpy()[:, 0]
            min_gap = min(min_gap, abs(x[1] - x[0]) - 0.2)
            crossed |= bool(x[0] > x[1])
        return run, min_gap, crossed

    def test_head_on_spheres_tunnel_without_mode(self):
        _, _, crossed = self._head_on_spheres(None)
        self.assertTrue(crossed, "baseline no longer tunnels; the regression below lost its discriminating power")

    def test_head_on_spheres_do_not_tunnel(self):
        run, min_gap, crossed = self._head_on_spheres(PenetrationFreeConfig(tolerance=_TOLERANCE))
        self.assertFalse(crossed)
        self.assertGreaterEqual(min_gap, -_TOLERANCE - _MEASURE_SLACK)
        self.assertEqual(run.failure_bits(), 0, run.data.status_messages())

    def _drop_all_shapes(self, ground: str, penetration_free):
        builder = newton.ModelBuilder()
        _add_ground(builder, ground)
        tilt = wp.quat_from_axis_angle(wp.normalize(wp.vec3(1.0, 0.3, 0.0)), 0.7)
        measures = []
        for i, kind in enumerate(_SHAPES):
            body = builder.add_body(xform=wp.transform((1.5 * i - 5.0, 0.0, 1.0), tilt))
            measures.append(_add_shape(builder, body, kind))
        run = _make_run(builder, penetration_free=penetration_free)
        qd = run.state_0.body_qd.numpy()
        qd[:, 2] = -60.0
        qd[:, 3:] = (3.0, 1.0, 0.0)
        run.state_0.body_qd.assign(qd)
        lowest = np.full(len(_SHAPES), np.inf)
        for _ in range(90):
            run.step()
            body_q = run.state_0.body_q.numpy()
            for i, (kind, params) in enumerate(measures):
                lowest[i] = min(lowest[i], _lowest_point(kind, params, body_q[i]))
        return run, lowest, run.state_0.body_q.numpy(), measures

    def _check_ground(self, ground: str):
        run, lowest, body_q, measures = self._drop_all_shapes(ground, PenetrationFreeConfig(tolerance=_TOLERANCE))
        for i, (kind, (measure, params)) in enumerate(zip(_SHAPES, measures, strict=True)):
            with self.subTest(shape=kind):
                self.assertGreaterEqual(lowest[i], -_TOLERANCE - _MEASURE_SLACK)
                # Bodies come to rest on the ground instead of stalling in the air.
                self.assertLess(body_q[i, 2], 0.35)
                if measure == "poly":
                    # Truncation must not hold a polyhedron balanced on an edge.
                    vertices_z = (params[0] @ _rotation_matrix(body_q[i, 3:]).T)[:, 2] + body_q[i, 2]
                    self.assertGreaterEqual(int(np.sum(vertices_z < 5.0e-3)), 3)
        self.assertEqual(run.failure_bits(), 0, run.data.status_messages())

    def test_shapes_on_plane(self):
        self._check_ground("plane")

    def test_shapes_on_box(self):
        self._check_ground("box")

    def test_shapes_on_mesh(self):
        self._check_ground("mesh")

    def test_shapes_on_heightfield(self):
        self._check_ground("heightfield")

    def test_shapes_tunnel_through_box_without_mode(self):
        _, lowest, _, _ = self._drop_all_shapes("box", None)
        self.assertLess(float(lowest.min()), -0.05, "baseline no longer penetrates the thin box")

    def test_overhanging_cube_tips_off(self):
        # A cube resting with its center of mass beyond its support's edge must
        # tip off. The face-to-face separating plane alone forbids the tipping,
        # so this needs per-update re-certification and room for a side at rest.
        for kind in ("box", "mesh"):
            with self.subTest(kind=kind):
                builder = newton.ModelBuilder()
                builder.add_ground_plane()
                for x, z in ((0.0, 0.1), (0.15, 0.3)):
                    body = builder.add_body(xform=wp.transform((x, 0.0, z), wp.quat_identity()))
                    if kind == "box":
                        builder.add_shape_box(body, hx=0.1, hy=0.1, hz=0.1)
                    else:
                        builder.add_shape_mesh(body, mesh=_box_mesh(0.1, 0.1, 0.1))
                run = _make_run(builder, penetration_free=PenetrationFreeConfig(), broad_phase="nxn")
                for _ in range(120):
                    run.step()
                body_q = run.state_0.body_q.numpy()
                self.assertLess(float(body_q[1, 2]), 0.105, "the overhanging cube did not land on the ground")
                self.assertEqual(run.failure_bits(), 0, run.data.status_messages())

    def test_leaning_mesh_tower_collapses(self):
        # Each cube overhangs the one below by less than half its width, but
        # the upper cubes' common center of mass lies beyond the base. Each
        # cube carries dozens of redundant triangle pairs, which an averaging
        # velocity limiter under-corrects until truncation locks the tower.
        builder = newton.ModelBuilder()
        builder.add_ground_plane()
        for level in range(5):
            body = builder.add_body(xform=wp.transform((0.06 * level, 0.0, 0.1 + 0.2 * level), wp.quat_identity()))
            builder.add_shape_mesh(body, mesh=_box_mesh(0.1, 0.1, 0.1))
        run = _make_run(builder, penetration_free=PenetrationFreeConfig(), broad_phase="nxn")
        for _ in range(240):
            run.step()
        self.assertLess(float(run.state_0.body_q.numpy()[:, 2].max()), 0.4, "the leaning tower is still standing")
        self.assertEqual(run.failure_bits(), 0, run.data.status_messages())

    def test_box_stack_settles(self):
        builder = newton.ModelBuilder()
        builder.add_ground_plane()
        half = 0.1
        bodies = []
        for level in range(5):
            body = builder.add_body(xform=wp.transform((0.0, 0.0, half + 2.0 * half * level), wp.quat_identity()))
            builder.add_shape_box(body, hx=half, hy=half, hz=half)
            bodies.append(body)
        run = _make_run(builder, penetration_free=PenetrationFreeConfig(tolerance=_TOLERANCE))
        for _ in range(120):
            run.step()
        body_q = run.state_0.body_q.numpy()
        body_qd = run.state_0.body_qd.numpy()
        expected = half + 2.0 * half * np.arange(5)
        np.testing.assert_allclose(body_q[:, 2], expected, atol=5.0 * _TOLERANCE)
        np.testing.assert_allclose(body_q[:, :2], 0.0, atol=1.0e-3)
        self.assertLess(float(np.max(np.abs(body_qd))), 0.05)
        self.assertEqual(run.failure_bits(), 0, run.data.status_messages())

    def test_kinematic_violation_is_reported(self):
        builder = newton.ModelBuilder(gravity=wp.vec3(0.0))
        builder.add_ground_plane()
        body = builder.add_body(xform=wp.transform((0.0, 0.0, 0.3), wp.quat_identity()), is_kinematic=True)
        builder.add_shape_box(body, hx=0.1, hy=0.1, hz=0.1)
        model = builder.finalize(device=require_cuda_graph_capture("Penetration-free PhoenX tests"))
        pipeline = newton.CollisionPipeline(
            model, contact_matching="sticky", penetration_free=PenetrationFreeConfig(tolerance=_TOLERANCE)
        )
        solver = SolverPhoenX(model, collision_pipeline=pipeline)
        contacts = pipeline.contacts()
        state_0, state_1 = model.state(), model.state()
        pipeline.collide(state_0, contacts, dt=_DT)
        # Kinematic bodies follow pose targets; drive this one through the ground.
        q = state_0.body_q.numpy()
        q[0, 2] = -0.3
        state_0.body_q.assign(q)
        solver.step(state_0, state_1, None, contacts, _DT)
        self.assertTrue(int(contacts._penetration_free.status.numpy()[0]) & int(STATUS_KINEMATIC_VIOLATION))

    def test_constraint_overflow_freezes_motion(self):
        builder = newton.ModelBuilder()
        builder.add_ground_plane()
        for i in range(4):
            body = builder.add_body(xform=wp.transform((0.5 * i, 0.0, 0.1), wp.quat_identity()))
            builder.add_shape_sphere(body, radius=0.1)
        run = _make_run(builder, penetration_free=PenetrationFreeConfig(max_constraints=2))
        before = run.state_0.body_q.numpy().copy()
        run.step()
        status = int(run.data.status.numpy()[0])
        self.assertTrue(status & int(STATUS_CONSTRAINT_OVERFLOW))
        np.testing.assert_allclose(run.state_0.body_q.numpy(), before, atol=1.0e-6)

    def test_teleported_state_freezes_motion(self):
        builder = newton.ModelBuilder()
        builder.add_ground_plane()
        body = builder.add_body(xform=wp.transform((0.0, 0.0, 0.5), wp.quat_identity()))
        builder.add_shape_sphere(body, radius=0.1)
        model = builder.finalize(device=require_cuda_graph_capture("Penetration-free PhoenX tests"))
        pipeline = newton.CollisionPipeline(
            model, contact_matching="sticky", penetration_free=PenetrationFreeConfig(tolerance=_TOLERANCE)
        )
        solver = SolverPhoenX(model, collision_pipeline=pipeline)
        contacts = pipeline.contacts()
        state_0, state_1 = model.state(), model.state()
        pipeline.collide(state_0, contacts, dt=_DT)
        q = state_0.body_q.numpy()
        q[0, 2] = -0.5  # teleport through the ground after detection
        state_0.body_q.assign(q)
        solver.step(state_0, state_1, None, contacts, _DT)
        self.assertTrue(int(contacts._penetration_free.status.numpy()[0]) & int(STATUS_REFERENCE_MISMATCH))
        np.testing.assert_allclose(state_1.body_q.numpy()[0, :3], q[0, :3], atol=1.0e-6)


if __name__ == "__main__":
    unittest.main()
