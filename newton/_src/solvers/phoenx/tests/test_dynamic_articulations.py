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
from newton._src.solvers.phoenx.dynamic import (
    DynamicsData,
    phoenx_articulation_body,
    phoenx_remove_articulation,
    phoenx_spawn_articulation,
)

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


@wp.kernel
def _spawn_kernel(
    dyn: DynamicsData,
    templates: wp.array[int],
    positions: wp.array[wp.vec3],
    velocity: wp.spatial_vector,
    out: wp.array[int],
):
    i = wp.tid()
    pose = wp.transform(positions[i], wp.quat_identity())
    out[i] = phoenx_spawn_articulation(dyn, templates[i], pose, velocity, 0)


@wp.kernel
def _remove_kernel(dyn: DynamicsData, instances: wp.array[int]):
    phoenx_remove_articulation(dyn, instances[wp.tid()])


@wp.kernel
def _bodies_kernel(dyn: DynamicsData, instance: int, out: wp.array[int]):
    i = wp.tid()
    out[i] = phoenx_articulation_body(dyn, instance, i)


class _Scene:
    def __init__(self, bodies: int, shapes: int, joints: int, articulations: int):
        builder = newton.ModelBuilder()
        builder.add_ground_plane()
        builder.reserve_dynamic_capacity(
            bodies=bodies,
            shapes=shapes,
            joints=joints,
            articulations=articulations,
            articulation_templates=[_chain_template(), _pendulum_template()],
        )
        self.model = builder.finalize()
        self.solver = newton.solvers.SolverPhoenX(
            self.model, joint_mode="maximal_pgs", step_layout="single_world", substeps=8, solver_iterations=8
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


if __name__ == "__main__":
    unittest.main()
