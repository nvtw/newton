# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Runtime body, shape, and joint creation in :class:`SolverPhoenX`.

Every scene creates and removes rows from device kernels inside the same CUDA graph that
collides and steps, so the tests cover the capturable end-to-end path.
"""

from __future__ import annotations

import ctypes
import unittest

import numpy as np
import warp as wp

import newton
from newton.solvers import phoenx

_HALF = 0.05
_DT = 1.0 / 60.0


@wp.func
def _add_box(dyn: phoenx.DynamicsData, position: wp.vec3, world: int) -> int:
    body_desc = phoenx.BodyDesc()
    body_desc.pose = wp.transform(position, wp.quat_identity())
    body_desc.mass = 1.0
    moment = 2.0 / 3.0 * _HALF * _HALF
    body_desc.inertia = wp.mat33(moment, 0.0, 0.0, 0.0, moment, 0.0, 0.0, 0.0, moment)
    body_desc.world = world
    body = phoenx.add_body(dyn, body_desc)
    if body < 0:
        return body
    shape = phoenx.ShapeDesc()
    shape.type = int(newton.GeoType.BOX)
    shape.transform = wp.transform_identity()
    shape.scale = wp.vec3(_HALF)
    shape.gap = 0.01
    shape.mu = 0.8
    shape.collision_group = 1
    shape.flags = int(newton.ShapeFlags.COLLIDE_SHAPES | newton.ShapeFlags.VISIBLE)
    shape.prototype = -1
    phoenx.add_shape(dyn, body, shape)
    return body


@wp.kernel
def _spawn_column_kernel(dyn: phoenx.DynamicsData, x: wp.array[float], z: float, out: wp.array[int]):
    """Spawn one box per entry of ``x`` once ``x`` is armed; each launch disarms it."""
    i = wp.tid()
    out[i] = _add_box(dyn, wp.vec3(x[i], 0.0, z), 0)


@wp.kernel
def _spawn_per_world_kernel(dyn: phoenx.DynamicsData, out: wp.array[int]):
    """One box per world, stacked at one spot; worlds must not collide with each other."""
    i = wp.tid()
    out[i] = _add_box(dyn, wp.vec3(0.0, 0.0, _HALF + 2.0 * _HALF * float(i)), i)


@wp.kernel
def _remove_bodies_kernel(dyn: phoenx.DynamicsData, bodies: wp.array[int]):
    phoenx.remove_body(dyn, bodies[wp.tid()])


@wp.kernel
def _pendulum_kernel(dyn: phoenx.DynamicsData, drive_angle: float, out: wp.array[int]):
    bob = _add_box(dyn, wp.vec3(0.5, 0.0, 2.0), 0)
    desc = phoenx.joint_desc(
        -1,
        bob,
        wp.transform(wp.vec3(0.0, 0.0, 2.0), wp.quat_identity()),
        wp.transform(wp.vec3(-0.5, 0.0, 0.0), wp.quat_identity()),
    )
    desc.angular_axes = wp.mat33(0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0)
    desc.angular_count = 1
    if drive_angle != 0.0:
        desc.target_mode[0] = int(newton.JointTargetMode.POSITION)
        desc.target_ke[0] = 2000.0
        desc.target_kd[0] = 50.0
    out[0] = bob
    out[1] = phoenx.add_joint(dyn, desc)


@wp.kernel
def _spawn_hull_kernel(dyn: phoenx.DynamicsData, prototype: int, scale: float, out: wp.array[int]):
    body_desc = phoenx.BodyDesc()
    body_desc.pose = wp.transform(wp.vec3(0.0, 0.0, 0.5), wp.quat_identity())
    body_desc.mass = 1.0
    body_desc.inertia = wp.identity(3, dtype=float) * 0.01
    body = phoenx.add_body(dyn, body_desc)
    shape = phoenx.ShapeDesc()
    shape.type = int(newton.GeoType.CONVEX_MESH)
    shape.transform = wp.transform_identity()
    shape.scale = wp.vec3(scale)
    shape.gap = 0.01
    shape.mu = 0.8
    shape.collision_group = 1
    shape.flags = int(newton.ShapeFlags.COLLIDE_SHAPES | newton.ShapeFlags.VISIBLE)
    shape.prototype = prototype
    out[0] = body
    out[1] = phoenx.add_shape(dyn, body, shape)


@wp.func
def _add_welded_box(dyn: phoenx.DynamicsData, out: wp.array[int]):
    """A box fixed to the world; a helper that forwards ``dyn`` one more level."""
    box = _add_box(dyn, wp.vec3(0.5, 0.0, 2.0), 0)
    out[0] = box
    out[1] = phoenx.add_joint(dyn, phoenx.joint_desc(-1, box, wp.transform_identity(), wp.transform_identity()))


@wp.kernel
def _spawn_and_remove_kernel(dyn: phoenx.DynamicsData, out: wp.array[int]):
    """Add a welded box and remove the body again before the next step commits it."""
    _add_welded_box(dyn, out)
    phoenx.remove_body(dyn, out[0])


@wp.kernel
def _remove_joint_kernel(dyn: phoenx.DynamicsData, joints: wp.array[int]):
    phoenx.remove_joint(dyn, joints[wp.tid()])


class _Scene:
    """Ground plane plus reserved rows, stepped by one captured collide + step graph."""

    def __init__(
        self, bodies: int, shapes: int, joints: int = 0, prototype: bool = False, worlds: int = 0, **solver_kwargs
    ):
        builder = newton.ModelBuilder()
        for _ in range(worlds):
            builder.begin_world()
            builder.end_world()
        builder.add_ground_plane()
        prototypes = []
        if prototype:
            hull = newton.Mesh.create_box(_HALF, duplicate_vertices=False)
            prototypes.append(builder.add_shape_convex_hull(-1, xform=wp.transform((5.0, 0.0, 0.0)), mesh=hull))
        self.prototype = prototypes[0] if prototypes else -1
        builder.reserve_dynamic_capacity(bodies=bodies, shapes=shapes, joints=joints, prototype_shapes=prototypes)
        self.model = builder.finalize()
        kwargs = {"joint_mode": "maximal_pgs", "step_layout": "single_world", "substeps": 4}
        kwargs.update(solver_kwargs)
        self.solver = newton.solvers.SolverPhoenX(self.model, **kwargs)
        self.dyn = self.solver.dynamics
        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()
        self.contacts = self.model._collision_pipeline.contacts()
        self.graph = None

    def _simulate(self):
        self.solver.collide(self.state_0, self.contacts)
        self.solver.step(self.state_0, self.state_1, self.control, self.contacts, _DT)
        wp.copy(self.state_0.body_q, self.state_1.body_q)
        wp.copy(self.state_0.body_qd, self.state_1.body_qd)

    def run(self, frames: int):
        if self.graph is None:
            with wp.ScopedCapture() as capture:
                self._simulate()
            self.graph = capture.graph
        for _ in range(frames):
            wp.capture_launch(self.graph)

    def body_q(self, bodies) -> np.ndarray:
        return self.state_0.body_q.numpy()[np.asarray(bodies)]


class TestDynamicsHandle(unittest.TestCase):
    def test_device_handle_stays_small(self):
        # Warp passes structs to functions by value; once a struct exceeds about 8 KB, forwarding
        # it through nested functions (a user helper calling add_body) copies it through local
        # memory on every call, which made each add_body cost tens of microseconds.
        self.assertLess(ctypes.sizeof(phoenx.DynamicsData.ctype), 8192)


@unittest.skipUnless(wp.is_cuda_available(), "runtime creation tests require CUDA graph capture")
class TestDynamicBodies(unittest.TestCase):
    def test_spawned_boxes_settle_on_ground(self):
        scene = _Scene(bodies=4, shapes=4)
        bodies = wp.zeros(4, dtype=int)
        x = wp.array([0.0, 0.3, 0.6, 0.9], dtype=float)
        wp.launch(_spawn_column_kernel, dim=4, inputs=[scene.dyn.data, x, 0.3, bodies])
        scene.run(90)
        z = scene.body_q(bodies.numpy())[:, 2]
        np.testing.assert_allclose(z, _HALF, atol=2.0e-3)
        self.assertEqual(int(scene.dyn.error_count.numpy()[0]), 0)

    def test_removed_support_drops_box_and_slot_is_reused(self):
        scene = _Scene(bodies=2, shapes=2)
        bottom = wp.zeros(1, dtype=int)
        top = wp.zeros(1, dtype=int)
        wp.launch(_spawn_column_kernel, dim=1, inputs=[scene.dyn.data, wp.array([0.0], dtype=float), _HALF, bottom])
        wp.launch(_spawn_column_kernel, dim=1, inputs=[scene.dyn.data, wp.array([0.0], dtype=float), 3 * _HALF, top])
        scene.run(60)
        self.assertAlmostEqual(float(scene.body_q(top.numpy())[0, 2]), 3 * _HALF, delta=3.0e-3)

        wp.launch(_remove_bodies_kernel, dim=1, inputs=[scene.dyn.data, bottom])
        scene.run(60)
        self.assertAlmostEqual(float(scene.body_q(top.numpy())[0, 2]), _HALF, delta=3.0e-3)
        # The removed slot was quarantined for one commit, then freed for reuse.
        np.testing.assert_array_equal(scene.dyn.bodies.data.counts.numpy()[:2], [1, 1])

        # The reused slot starts clean far from its previous contacts.
        again = wp.zeros(1, dtype=int)
        wp.launch(_spawn_column_kernel, dim=1, inputs=[scene.dyn.data, wp.array([1.0], dtype=float), _HALF, again])
        self.assertEqual(int(again.numpy()[0]), int(bottom.numpy()[0]))
        scene.run(30)
        q = scene.body_q(again.numpy())[0]
        np.testing.assert_allclose(q[:3], [1.0, 0.0, _HALF], atol=3.0e-3)
        qd = scene.state_0.body_qd.numpy()[again.numpy()[0]]
        self.assertLess(float(np.linalg.norm(qd)), 1.0e-2)

    def test_convex_hull_from_prototype_is_rescaled(self):
        scene = _Scene(bodies=1, shapes=1, prototype=True)
        ids = wp.zeros(2, dtype=int)
        wp.launch(_spawn_hull_kernel, dim=1, inputs=[scene.dyn.data, scene.prototype, 2.0, ids])
        scene.run(90)
        self.assertAlmostEqual(float(scene.body_q([ids.numpy()[0]])[0, 2]), 2.0 * _HALF, delta=3.0e-3)
        shape = int(ids.numpy()[1])
        np.testing.assert_allclose(scene.model.shape_collision_aabb_upper.numpy()[shape], 2.0 * _HALF, rtol=1.0e-5)
        # The prototype only lends geometry.
        self.assertFalse(scene.model.shape_flags.numpy()[scene.prototype] & int(newton.ShapeFlags.COLLIDE_SHAPES))

    def test_runtime_shapes_respect_worlds(self):
        scene = _Scene(bodies=2, shapes=2, worlds=2)
        ids = wp.zeros(2, dtype=int)
        wp.launch(_spawn_per_world_kernel, dim=2, inputs=[scene.dyn.data, ids])
        scene.run(60)
        np.testing.assert_allclose(scene.body_q(ids.numpy())[:, 2], _HALF, atol=3.0e-3)

    def test_full_pool_reports_errors(self):
        scene = _Scene(bodies=2, shapes=2)
        bodies = wp.zeros(3, dtype=int)
        wp.launch(
            _spawn_column_kernel, dim=3, inputs=[scene.dyn.data, wp.array([0.0, 0.3, 0.6], dtype=float), 0.3, bodies]
        )
        self.assertEqual(int(np.count_nonzero(bodies.numpy() < 0)), 1)
        self.assertEqual(int(scene.dyn.error_count.numpy()[0]), 1)

    def test_runtime_revolute_holds_hinge_and_drive(self):
        scene = _Scene(bodies=1, shapes=1, joints=1)
        ids = wp.zeros(2, dtype=int)
        wp.launch(_pendulum_kernel, dim=1, inputs=[scene.dyn.data, 0.0, ids])
        scene.run(60)
        bob = scene.body_q([ids.numpy()[0]])[0]
        rotation = wp.quat(*bob[3:].tolist())
        anchor = np.asarray(bob[:3]) + np.asarray(wp.quat_rotate(rotation, wp.vec3(-0.5, 0.0, 0.0)))
        np.testing.assert_allclose(anchor, [0.0, 0.0, 2.0], atol=5.0e-3)
        self.assertLess(float(bob[2]), 1.9)  # swinging

        # A position drive to the authored zero angle holds the bob horizontal.
        driven = _Scene(bodies=1, shapes=1, joints=1)
        wp.launch(_pendulum_kernel, dim=1, inputs=[driven.dyn.data, 1.0, ids])
        driven.run(60)
        self.assertAlmostEqual(float(driven.body_q([ids.numpy()[0]])[0, 2]), 2.0, delta=0.05)

        # Removing the joint releases the bob.
        wp.launch(_remove_joint_kernel, dim=1, inputs=[driven.dyn.data, ids[1:]])
        driven.run(30)
        self.assertLess(float(driven.body_q([ids.numpy()[0]])[0, 2]), 1.7)

    def test_body_removal_cascades_to_shapes_and_joints(self):
        scene = _Scene(bodies=1, shapes=1, joints=1)
        ids = wp.zeros(2, dtype=int)
        wp.launch(_pendulum_kernel, dim=1, inputs=[scene.dyn.data, 0.0, ids])
        scene.run(5)
        wp.launch(_remove_bodies_kernel, dim=1, inputs=[scene.dyn.data, ids[:1]])
        scene.run(3)
        for pool in (scene.dyn.bodies, scene.dyn.shapes, scene.dyn.joints):
            np.testing.assert_array_equal(pool.data.counts.numpy()[:2], [0, 1])

    def test_rows_removed_before_their_commit_are_never_written(self):
        scene = _Scene(bodies=1, shapes=1, joints=1)
        ids = wp.zeros(2, dtype=int)
        wp.launch(_spawn_and_remove_kernel, dim=1, inputs=[scene.dyn.data, ids])
        mass = scene.model.body_mass.numpy()[ids.numpy()[0]]
        scene.run(2)  # cancelled, then out of quarantine
        for pool in (scene.dyn.bodies, scene.dyn.shapes, scene.dyn.joints):
            np.testing.assert_array_equal(pool.data.counts.numpy()[:2], [0, 1])
        self.assertEqual(scene.model.body_mass.numpy()[ids.numpy()[0]], mass)
        self.assertEqual(int(scene.dyn.error_count.numpy()[0]), 0)

        # The slots are handed out again and simulate normally.
        bodies = wp.zeros(1, dtype=int)
        wp.launch(_spawn_column_kernel, dim=1, inputs=[scene.dyn.data, wp.array([0.0], dtype=float), 0.3, bodies])
        self.assertEqual(int(bodies.numpy()[0]), int(ids.numpy()[0]))
        scene.run(90)
        self.assertAlmostEqual(float(scene.body_q(bodies.numpy())[0, 2]), _HALF, delta=3.0e-3)

    def test_unsupported_configurations_raise(self):
        builder = newton.ModelBuilder()
        builder.add_ground_plane()
        builder.reserve_dynamic_capacity(bodies=1, shapes=1, joints=1)
        model = builder.finalize()
        with self.assertRaises(ValueError):
            newton.CollisionPipeline(model, broad_phase="explicit")
        with self.assertRaises(ValueError):
            newton.solvers.SolverPhoenX(model, joint_mode="maximal_direct")

    def test_runtime_body_falls_in_a_reduced_articulation_scene(self):
        """Reduced scenes still integrate runtime bodies, whose rows start static."""
        builder = newton.ModelBuilder()
        builder.add_ground_plane()
        link = builder.add_link(xform=wp.transform(wp.vec3(3.0, 0.0, 0.2), wp.quat_identity()))
        builder.add_shape_box(link, hx=0.2, hy=0.2, hz=0.2)
        builder.add_articulation([builder.add_joint_free(link)])
        builder.reserve_dynamic_capacity(bodies=1, shapes=1)
        model = builder.finalize()
        solver = newton.solvers.SolverPhoenX(model, joint_mode="reduced")
        bodies = wp.zeros(1, dtype=int)
        wp.launch(_spawn_column_kernel, dim=1, inputs=[solver.dynamics.data, wp.array([0.0], dtype=float), 2.0, bodies])
        state_0, state_1, control = model.state(), model.state(), model.control()
        contacts = model._collision_pipeline.contacts()
        with wp.ScopedCapture() as capture:
            solver.collide(state_0, contacts)
            solver.step(state_0, state_1, control, contacts, _DT)
            for name in ("body_q", "body_qd", "joint_q", "joint_qd"):
                wp.copy(getattr(state_0, name), getattr(state_1, name))
        for _ in range(90):
            wp.capture_launch(capture.graph)
        np.testing.assert_allclose(state_0.body_q.numpy()[bodies.numpy()[0]][2], _HALF, atol=5.0e-3)


if __name__ == "__main__":
    unittest.main()
