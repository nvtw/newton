# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Runtime articulation spawning from templates in :class:`SolverPhoenX`.

All templates draw on one shared reserve of body, shape, and joint rows. Spawns and removals
run in device kernels next to the captured collide + step graph.
"""

from __future__ import annotations

import itertools
import unittest

import numpy as np
import warp as wp

import newton
from newton.solvers import phoenx

_DT = 1.0 / 60.0
_LINK = 0.2  # half length of a chain link along x [m]
_RADIUS = 0.05
_CHAIN, _PENDULUM = 0, 1


def _chain_template(links: int = 4) -> newton.ModelBuilder:
    """Free-floating capsule chain hinged about y; the capsules overlap at every hinge."""
    builder = newton.ModelBuilder()
    capsule = wp.quat_from_axis_angle(wp.vec3(0.0, 1.0, 0.0), 0.5 * wp.pi)  # capsule axis z -> x
    previous = -1
    for i in range(links):
        x = 2.0 * _LINK * i
        pose = wp.transform(wp.vec3(x, 0.0, 0.0), wp.quat_identity())
        body = builder.add_body(xform=pose) if i == 0 else builder.add_link(xform=pose)
        builder.add_shape_capsule(body, xform=wp.transform((0.0, 0.0, 0.0), capsule), radius=_RADIUS, half_height=_LINK)
        if previous >= 0:
            builder.add_joint_revolute(
                previous,
                body,
                parent_xform=wp.transform(wp.vec3(_LINK, 0.0, 0.0), wp.quat_identity()),
                child_xform=wp.transform(wp.vec3(-_LINK, 0.0, 0.0), wp.quat_identity()),
                axis=newton.Axis.Y,
            )
        previous = body
    return builder


def _rope_template(links: int = 4) -> newton.ModelBuilder:
    """The capsule chain of :func:`_chain_template` with ball joints."""
    builder = newton.ModelBuilder()
    capsule = wp.quat_from_axis_angle(wp.vec3(0.0, 1.0, 0.0), 0.5 * wp.pi)
    previous = -1
    for i in range(links):
        body = builder.add_link(xform=wp.transform(wp.vec3(2.0 * _LINK * i, 0.0, 0.0), wp.quat_identity()))
        builder.add_shape_capsule(body, xform=wp.transform((0.0, 0.0, 0.0), capsule), radius=_RADIUS, half_height=_LINK)
        if previous >= 0:
            builder.add_joint_ball(
                previous,
                body,
                parent_xform=wp.transform(wp.vec3(_LINK, 0.0, 0.0), wp.quat_identity()),
                child_xform=wp.transform(wp.vec3(-_LINK, 0.0, 0.0), wp.quat_identity()),
            )
        previous = body
    return builder


def _pendulum_template() -> newton.ModelBuilder:
    """Box hinged to the world 0.5 m along -x of its template origin."""
    builder = newton.ModelBuilder()
    body = builder.add_link(xform=wp.transform(wp.vec3(0.5, 0.0, 0.0), wp.quat_identity()))
    builder.add_shape_box(body, hx=0.05, hy=0.05, hz=0.05)
    builder.add_joint_revolute(
        -1,
        body,
        parent_xform=wp.transform(wp.vec3(0.0, 0.0, 0.0), wp.quat_identity()),
        child_xform=wp.transform(wp.vec3(-0.5, 0.0, 0.0), wp.quat_identity()),
        axis=newton.Axis.Y,
    )
    return builder


def _hanging_chain_template(links: int = 12) -> newton.ModelBuilder:
    """Chain hinged to the world with a last link 50 times heavier than the others."""
    builder = newton.ModelBuilder()
    previous = -1
    for i in range(links):
        body = builder.add_link(xform=wp.transform(wp.vec3(2.0 * _LINK * i + _LINK, 0.0, 0.0), wp.quat_identity()))
        density = 1000.0 * (50.0 if i == links - 1 else 1.0)
        builder.add_shape_box(
            body, hx=0.9 * _LINK, hy=0.03, hz=0.03, cfg=newton.ModelBuilder.ShapeConfig(density=density)
        )
        anchor = wp.vec3(0.0, 0.0, 0.0) if previous < 0 else wp.vec3(_LINK, 0.0, 0.0)
        builder.add_joint_revolute(
            previous,
            body,
            parent_xform=wp.transform(anchor, wp.quat_identity()),
            child_xform=wp.transform(wp.vec3(-_LINK, 0.0, 0.0), wp.quat_identity()),
            axis=newton.Axis.Y,
        )
        previous = body
    return builder


@wp.kernel
def _remove_joint_kernel(dyn: phoenx.DynamicsData, instance: int, index: int):
    phoenx.remove_joint(dyn, phoenx.articulation_joint(dyn, instance, index))


@wp.kernel
def _spawn_kernel(
    dyn: phoenx.DynamicsData,
    templates: wp.array[int],
    positions: wp.array[wp.vec3],
    velocity: wp.spatial_vector,
    out: wp.array[int],
):
    i = wp.tid()
    pose = wp.transform(positions[i], wp.quat_identity())
    out[i] = phoenx.add_articulation(dyn, templates[i], pose, velocity, 0)


@wp.kernel
def _remove_kernel(dyn: phoenx.DynamicsData, instances: wp.array[int]):
    phoenx.remove_articulation(dyn, instances[wp.tid()])


@wp.kernel
def _bodies_kernel(dyn: phoenx.DynamicsData, instance: int, out: wp.array[int]):
    i = wp.tid()
    out[i] = phoenx.articulation_body(dyn, instance, i)


class _Scene:
    def __init__(self, bodies: int, shapes: int, joints: int, articulations: int, templates=None):
        builder = newton.ModelBuilder()
        builder.add_ground_plane()
        builder.reserve_dynamic_capacity(
            bodies=bodies,
            shapes=shapes,
            joints=joints,
            articulations=articulations,
            articulation_templates=templates or [_chain_template(), _pendulum_template()],
        )
        self.model = builder.finalize()
        self.solver = newton.solvers.SolverPhoenX(
            self.model, joint_mode="maximal_pgs", step_layout="single_world", substeps=4, solver_iterations=4
        )
        self.dyn = self.solver.dynamics
        self.state_0, self.state_1 = self.model.state(), self.model.state()
        self.control = self.model.control()
        self.contacts = self.model._collision_pipeline.contacts()
        with wp.ScopedCapture() as capture:
            self.solver.collide(self.state_0, self.contacts)
            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, _DT)
            wp.copy(self.state_0.body_q, self.state_1.body_q)
            wp.copy(self.state_0.body_qd, self.state_1.body_qd)
        self.graph = capture.graph

    def spawn(self, templates, positions, velocity=None) -> np.ndarray:
        velocity = wp.spatial_vector() if velocity is None else velocity
        out = wp.zeros(len(templates), dtype=int)
        wp.launch(
            _spawn_kernel,
            dim=len(templates),
            inputs=[self.dyn.data, wp.array(templates, dtype=int), wp.array(positions, dtype=wp.vec3), velocity, out],
        )
        return out.numpy()

    def remove(self, instances):
        wp.launch(_remove_kernel, dim=len(instances), inputs=[self.dyn.data, wp.array(instances, dtype=int)])

    def bodies(self, instance: int, count: int) -> np.ndarray:
        out = wp.zeros(count, dtype=int)
        wp.launch(_bodies_kernel, dim=count, inputs=[self.dyn.data, int(instance), out])
        return out.numpy()

    def run(self, frames: int):
        for _ in range(frames):
            wp.capture_launch(self.graph)

    def free_counts(self) -> list[int]:
        return [int(pool.data.counts.numpy()[1]) for pool in (self.dyn.bodies, self.dyn.shapes, self.dyn.joints)]


def _hinge_gap(body_q: np.ndarray, a: int, b: int) -> float:
    """Distance between the hinge points of consecutive chain links [m]."""
    pa = wp.transform_point(wp.transform(*body_q[a].tolist()), wp.vec3(_LINK, 0.0, 0.0))
    pb = wp.transform_point(wp.transform(*body_q[b].tolist()), wp.vec3(-_LINK, 0.0, 0.0))
    return float(np.linalg.norm(np.asarray(pa) - np.asarray(pb)))


@unittest.skipUnless(wp.is_cuda_available(), "runtime creation tests require CUDA graph capture")
class TestDynamicArticulations(unittest.TestCase):
    def test_spawned_chains_keep_their_hinges_and_rest(self):
        scene = _Scene(bodies=8, shapes=8, joints=6, articulations=2)
        instances = scene.spawn(
            [_CHAIN, _CHAIN], [(0.0, 0.0, 0.5), (0.0, 1.0, 0.8)], wp.spatial_vector(0.5, 0, 0, 0, 0, 1)
        )
        self.assertTrue(np.all(instances >= 0))
        scene.run(120)
        q = scene.state_0.body_q.numpy()
        self.assertTrue(np.isfinite(q).all())
        for instance in instances:
            links = scene.bodies(instance, 4)
            self.assertTrue(np.all(links >= 0))
            for a, b in itertools.pairwise(links):
                self.assertLess(_hinge_gap(q, a, b), 5.0e-3)
            # Resting on the ground; overlapping capsules of one chain never pushed each other apart.
            np.testing.assert_allclose(q[links, 2], _RADIUS, atol=1.0e-2)
        self.assertEqual(int(scene.dyn.error_count.numpy()[0]), 0)

    def test_templates_share_rows_and_failed_spawns_roll_back(self):
        # Rows for exactly one chain and one pendulum.
        scene = _Scene(bodies=5, shapes=5, joints=4, articulations=4)
        instances = scene.spawn([_CHAIN, _PENDULUM], [(0, 0, 0.5), (0, 2, 1.5)])
        self.assertTrue(np.all(instances >= 0))
        self.assertLess(scene.spawn([_PENDULUM], [(0, 3, 1.5)])[0], 0)
        self.assertGreater(int(scene.dyn.error_count.numpy()[0]), 0)
        scene.run(2)
        self.assertEqual(scene.free_counts(), [0, 0, 0])

        # Removing the chain frees rows for four more pendulums.
        scene.remove([int(instances[0])])
        scene.run(2)
        self.assertEqual(scene.free_counts(), [4, 4, 3])
        more = scene.spawn([_PENDULUM] * 3, [(1.0, float(i), 1.5) for i in range(3)])
        self.assertTrue(np.all(more >= 0))
        scene.run(30)
        self.assertTrue(np.isfinite(scene.state_0.body_q.numpy()).all())

    def test_anchored_template_joint_follows_the_spawn_pose(self):
        scene = _Scene(bodies=4, shapes=4, joints=3, articulations=1)
        spawn = np.array([1.0, 2.0, 1.5])
        pivot = spawn - [0.5, 0.0, 0.0]  # templates are posed by their first body, the bob
        instance = scene.spawn([_PENDULUM], [tuple(spawn)])[0]
        scene.run(20)  # a quarter swing from horizontal
        bob = scene.bodies(instance, 1)[0]
        q = scene.state_0.body_q.numpy()[bob]
        anchor = wp.transform_point(wp.transform(*q.tolist()), wp.vec3(-0.5, 0.0, 0.0))
        np.testing.assert_allclose(np.asarray(anchor), pivot, atol=5.0e-3)
        self.assertLess(float(q[2]), pivot[2] - 0.3)  # swung down

    def test_spawned_trees_are_solved_exactly(self):
        # Local joint blocks open the hinges of this whipping chain by about 10 cm.
        scene = _Scene(bodies=12, shapes=12, joints=12, articulations=1, templates=[_hanging_chain_template()])
        instance = scene.spawn([0], [(_LINK, 0.0, 4.0)])[0]
        links = scene.bodies(instance, 12)
        worst = 0.0
        for _ in range(120):
            scene.run(1)
            q = scene.state_0.body_q.numpy()
            worst = max(worst, *(_hinge_gap(q, a, b) for a, b in itertools.pairwise(links)))
        self.assertLess(worst, 0.03)

    def test_removing_a_tree_joint_falls_back_to_local_blocks(self):
        scene = _Scene(bodies=12, shapes=12, joints=12, articulations=1, templates=[_hanging_chain_template()])
        instance = scene.spawn([0], [(_LINK, 0.0, 4.0)])[0]
        scene.run(20)
        self.assertEqual(int(scene.solver.dynamics.data.tree_body_count.numpy()[instance]), 12)
        wp.launch(_remove_joint_kernel, dim=1, inputs=[scene.dyn.data, int(instance), 6])
        self.assertEqual(int(scene.solver.dynamics.data.tree_body_count.numpy()[instance]), 0)
        scene.run(60)
        q = scene.state_0.body_q.numpy()[scene.bodies(instance, 12)]
        self.assertTrue(np.isfinite(q).all())
        self.assertLess(float(q[-1, 2]), 1.0)  # the lower half fell
        self.assertEqual(int(scene.dyn.error_count.numpy()[0]), 0)


def _ragdoll_template() -> newton.ModelBuilder:
    """Torso with four ball-jointed limbs one level below the root."""
    builder = newton.ModelBuilder()
    torso = builder.add_link()
    builder.add_shape_box(torso, hx=0.15, hy=0.1, hz=0.25)
    for k, (x, z) in enumerate(((0.25, 0.1), (-0.25, 0.1), (0.1, -0.45), (-0.1, -0.45))):
        limb = builder.add_link(xform=wp.transform(wp.vec3(x, 0.0, z), wp.quat_identity()))
        builder.add_shape_box(limb, hx=0.05, hy=0.05, hz=0.15)
        anchor = wp.vec3(x, 0.0, z + (0.15 if k >= 2 else 0.0))
        builder.add_joint_ball(
            torso,
            limb,
            parent_xform=wp.transform(anchor, wp.quat_identity()),
            child_xform=wp.transform(anchor - wp.vec3(x, 0.0, z), wp.quat_identity()),
        )
    return builder


@wp.kernel
def _spawn_posed_kernel(dyn: phoenx.DynamicsData, templates: wp.array[int], x: wp.array[float], out: wp.array[int]):
    i = wp.tid()
    pose = wp.transform(wp.vec3(x[i], 0.0, 1.5), wp.quat_from_axis_angle(wp.vec3(0.0, 1.0, 0.0), 0.4))
    out[i] = phoenx.add_articulation(dyn, templates[i], pose, wp.spatial_vector(0.0, 0.0, 0.0, 0.0, 0.0, 1.0), 0)


class _ReducedScene:
    """Generalized-coordinate instance slots, stepped like any reduced scene in one captured graph."""

    def __init__(self, slots: int | list[int], templates):
        builder = newton.ModelBuilder()
        builder.add_ground_plane()
        builder.reserve_dynamic_capacity(
            shapes=int(np.sum(slots)) * 16,
            articulations=slots,
            articulation_templates=templates,
            articulation_coordinates="reduced",
        )
        self.model = builder.finalize()
        self.capacity = self.model._dynamic_capacity
        self.solver = newton.solvers.SolverPhoenX(self.model, joint_mode="reduced", substeps=4, solver_iterations=4)
        self.dyn = self.solver.dynamics
        self.state_0, self.state_1 = self.model.state(), self.model.state()
        self.control = self.model.control()
        self.contacts = self.model._collision_pipeline.contacts()
        with wp.ScopedCapture() as capture:
            self.solver.collide(self.state_0, self.contacts)
            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, _DT)
            for name in ("body_q", "body_qd", "joint_q", "joint_qd"):
                wp.copy(getattr(self.state_0, name), getattr(self.state_1, name))
        self.graph = capture.graph

    def spawn(self, templates, xs) -> np.ndarray:
        out = wp.zeros(len(templates), dtype=int)
        wp.launch(
            _spawn_posed_kernel,
            dim=len(templates),
            inputs=[self.dyn.data, wp.array(templates, dtype=int), wp.array(xs, dtype=float), out],
        )
        return out.numpy()

    def run(self, frames: int):
        for _ in range(frames):
            wp.capture_launch(self.graph)

    def joint_gap(self) -> float:
        """Largest joint opening of the live instances [m]."""
        q = self.state_0.body_q.numpy()
        reduced = self.solver._reduced_articulation.bodies.reduced
        starts, ends = reduced.articulation_start.numpy(), reduced.articulation_end.numpy()
        parent, child = self.model.joint_parent.numpy(), self.model.joint_child.numpy()
        xp, xc = self.model.joint_X_p.numpy(), self.model.joint_X_c.numpy()
        worst = 0.0
        for a in range(len(ends)):
            for j in range(starts[a] + 1, ends[a]):
                pa = wp.transform_point(wp.transform(*q[parent[j]].tolist()), wp.vec3(*xp[j][:3]))
                pb = wp.transform_point(wp.transform(*q[child[j]].tolist()), wp.vec3(*xc[j][:3]))
                worst = max(worst, float(np.linalg.norm(np.asarray(pa) - np.asarray(pb))))
        return worst

    def slot_bodies(self, slot: int) -> np.ndarray:
        start = self.capacity.slot_body_start + slot * self.capacity.slot_size
        return self.state_0.body_q.numpy()[start : start + self.capacity.slot_size]


@unittest.skipUnless(wp.is_cuda_available(), "runtime creation tests require CUDA graph capture")
class TestReducedRuntimeArticulations(unittest.TestCase):
    def test_spawned_trees_fall_and_rest_with_exact_joints(self):
        scene = _ReducedScene(3, [_ragdoll_template(), _chain_template(), _spider_template()])
        instances = scene.spawn([0, 1, 2], [-3.0, 0.0, 3.0])
        np.testing.assert_array_equal(instances, [0, 1, 2])
        scene.run(150)
        for slot, bodies in enumerate((5, 4, 9)):
            q = scene.slot_bodies(slot)[:bodies]
            self.assertTrue(np.isfinite(q).all())
            self.assertLess(float(q[:, 2].max()), 0.6, f"slot {slot} did not come to rest on the ground")
            self.assertGreater(float(q[:, 2].min()), 0.0, f"slot {slot} sank through the ground")
        self.assertLess(scene.joint_gap(), 1.0e-4)  # generalized coordinates keep joints closed exactly
        self.assertEqual(int(scene.dyn.error_count.numpy()[0]), 0)

    def test_slots_are_reused_by_other_templates(self):
        # Two slots sized for the spider; a spider fills one exactly.
        scene = _ReducedScene(2, [_ragdoll_template(), _spider_template()])
        scene.spawn([1, 0], [-2.0, 2.0])
        scene.run(40)
        wp.launch(_remove_kernel, dim=2, inputs=[scene.dyn.data, wp.array([0, 1], dtype=int)])
        scene.run(2)  # the slots pass quarantine
        parked = scene.slot_bodies(0).copy()
        scene.run(10)
        np.testing.assert_allclose(scene.slot_bodies(0), parked)  # empty slots are inert
        np.testing.assert_array_equal(scene.spawn([0, 1], [-2.0, 2.0]), [0, 1])
        scene.run(120)
        for slot in range(2):
            self.assertTrue(np.isfinite(scene.slot_bodies(slot)).all())
        self.assertLess(scene.joint_gap(), 1.0e-4)
        self.assertEqual(int(scene.dyn.error_count.numpy()[0]), 0)

    def test_spinning_tree_moves_alike_in_a_reused_and_a_fresh_slot(self):
        # A rope spinning about its axis takes the midpoint step, which fills the slot's
        # ball-joint rates. A hinge chain spawned into that slot later pads those rates and
        # must move exactly like the same chain in a fresh slot.
        scene = _ReducedScene(2, [_rope_template(), _chain_template()])
        spin = wp.spatial_vector(0.0, 0.0, 0.0, 60.0, 0.0, 3.0)
        positions = wp.array([wp.vec3(-2.0, 0.0, 2.0), wp.vec3(2.0, 0.0, 2.0)], dtype=wp.vec3)
        instances = wp.zeros(2, dtype=int)
        wp.launch(_spawn_kernel, dim=1, inputs=[scene.dyn.data, wp.array([0], dtype=int), positions, spin, instances])
        scene.run(5)
        wp.launch(_remove_kernel, dim=1, inputs=[scene.dyn.data, wp.array([0], dtype=int)])
        scene.run(2)  # the slot passes quarantine
        wp.launch(
            _spawn_kernel, dim=2, inputs=[scene.dyn.data, wp.array([1, 1], dtype=int), positions, spin, instances]
        )
        np.testing.assert_array_equal(instances.numpy(), [0, 1])
        scene.run(20)
        midpoint = scene.solver._reduced_articulation.system.midpoint.flag.numpy()
        self.assertTrue(midpoint[-2:].all())  # both chains take the midpoint step
        reused, fresh = scene.slot_bodies(0)[:4], scene.slot_bodies(1)[:4]
        self.assertTrue(np.isfinite(reused).all())
        self.assertGreater(float(fresh[:, 2].min()), 0.5)  # still in free fall
        np.testing.assert_allclose(reused[:, :3] - reused[0, :3], fresh[:, :3] - fresh[0, :3], atol=1.0e-4)
        np.testing.assert_allclose(reused[:, 3:], fresh[:, 3:], atol=1.0e-4)


def _spider_template() -> newton.ModelBuilder:
    """A hub with three two-segment legs on hinges: nine bodies, depth three."""
    builder = newton.ModelBuilder()
    hub = builder.add_link()
    builder.add_shape_sphere(hub, radius=0.12)
    for k in range(4):
        yaw = wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), 0.5 * wp.pi * k)
        out = wp.quat_rotate(yaw, wp.vec3(1.0, 0.0, 0.0))
        parent, base = hub, 0.12
        for segment in range(2):
            body = builder.add_link(xform=wp.transform(out * (base + 0.12), yaw))
            builder.add_shape_box(body, hx=0.1, hy=0.03, hz=0.03)
            builder.add_joint_revolute(
                parent,
                body,
                parent_xform=wp.transform(
                    out * base if segment == 0 else wp.vec3(0.12, 0.0, 0.0), yaw if segment == 0 else wp.quat_identity()
                ),
                child_xform=wp.transform(wp.vec3(-0.12, 0.0, 0.0), wp.quat_identity()),
                axis=newton.Axis.Y,
            )
            parent, base = body, base + 0.24
    return builder


@unittest.skipUnless(wp.is_cuda_available(), "runtime creation tests require CUDA graph capture")
class TestBuiltRuntimeArticulations(unittest.TestCase):
    """Instances reserved per template keep their joint tree, which the fused substep solves."""

    @staticmethod
    def _bodies(scene: _ReducedScene, instance: int, count: int) -> np.ndarray:
        start = int(scene.dyn.data.instance_body_start.numpy()[instance])
        return scene.state_0.body_q.numpy()[start : start + count]

    def test_instances_fall_and_rest_on_the_fused_substep(self):
        scene = _ReducedScene([2, 1, 2], [_ragdoll_template(), _chain_template(), _spider_template()])
        self.assertIsNotNone(scene.solver._reduced_articulation.fused_substep)
        instances = scene.spawn([0, 1, 2, 2], [-3.0, 0.0, 3.0, 6.0])
        np.testing.assert_array_equal(instances, [0, 2, 3, 4])  # each template draws on its own instances
        scene.run(150)
        for instance, bodies in zip(instances, (5, 4, 9, 9), strict=True):
            q = self._bodies(scene, instance, bodies)
            self.assertTrue(np.isfinite(q).all())
            self.assertLess(float(q[:, 2].max()), 0.6, f"instance {instance} did not come to rest on the ground")
            self.assertGreater(float(q[:, 2].min()), 0.0, f"instance {instance} sank through the ground")
        self.assertLess(float(self._bodies(scene, 1, 5)[:, 2].max()), -1.0e3)  # the spare ragdoll stays parked
        self.assertLess(scene.joint_gap(), 1.0e-4)
        self.assertEqual(int(scene.dyn.error_count.numpy()[0]), 0)

    def test_removed_instance_is_inert_and_only_its_template_reuses_it(self):
        scene = _ReducedScene([1, 1], [_chain_template(), _spider_template()])
        scene.spawn([0, 1], [-2.0, 2.0])
        scene.run(60)
        contacts = int(scene.contacts.rigid_contact_count.numpy()[0])
        wp.launch(_remove_kernel, dim=1, inputs=[scene.dyn.data, wp.array([1], dtype=int)])
        scene.run(2)  # the instance passes quarantine
        left = self._bodies(scene, 1, 9).copy()
        scene.run(10)
        np.testing.assert_array_equal(self._bodies(scene, 1, 9), left)
        self.assertLess(int(scene.contacts.rigid_contact_count.numpy()[0]), contacts)  # its shapes are gone
        # The chain's only instance is in the scene; the spider's is free again.
        np.testing.assert_array_equal(scene.spawn([0, 1], [5.0, 5.0]), [-1, 1])
        self.assertEqual(int(scene.dyn.error_count.numpy()[0]), 1)
        scene.run(120)
        q = self._bodies(scene, 1, 9)
        self.assertTrue(np.isfinite(q).all())
        self.assertLess(float(q[:, 2].max()), 0.6)
        self.assertGreater(float(q[:, 2].min()), 0.0)
        self.assertLess(scene.joint_gap(), 1.0e-4)

    def test_respawned_instance_moves_like_a_fresh_one(self):
        def fall(respawn: bool) -> np.ndarray:
            scene = _ReducedScene([1], [_ragdoll_template()])
            if respawn:
                scene.spawn([0], [1.0])
                scene.run(40)  # it lands and holds contacts
                wp.launch(_remove_kernel, dim=1, inputs=[scene.dyn.data, wp.array([0], dtype=int)])
                scene.run(2)
            np.testing.assert_array_equal(scene.spawn([0], [-2.0]), [0])
            scene.run(60)
            return self._bodies(scene, 0, 5).copy()

        np.testing.assert_array_equal(fall(True), fall(False))

    def test_anchored_instance_follows_the_spawn_pose(self):
        scene = _ReducedScene([1], [_pendulum_template()])
        self.assertIsNotNone(scene.solver._reduced_articulation.fused_substep)
        scene.spawn([0], [1.0])
        pose = wp.transform(wp.vec3(1.0, 0.0, 1.5), wp.quat_from_axis_angle(wp.vec3(0.0, 1.0, 0.0), 0.4))
        pivot = np.asarray(wp.transform_point(pose, wp.vec3(-0.5, 0.0, 0.0)))
        scene.run(20)
        q = self._bodies(scene, 0, 1)[0]
        anchor = wp.transform_point(wp.transform(*q.tolist()), wp.vec3(-0.5, 0.0, 0.0))
        np.testing.assert_allclose(np.asarray(anchor), pivot, atol=1.0e-4)
        self.assertLess(float(q[2]), pivot[2] - 0.3)  # swung down

    def test_counts_per_template_are_validated(self):
        for coordinates, counts in (("maximal", [1]), ("reduced", [1, 1]), ("reduced", [-1])):
            builder = newton.ModelBuilder()
            with self.assertRaises(ValueError):
                builder.reserve_dynamic_capacity(
                    bodies=8,
                    shapes=8,
                    joints=8,
                    articulations=counts,
                    articulation_templates=[_chain_template()],
                    articulation_coordinates=coordinates,
                )


if __name__ == "__main__":
    unittest.main()
