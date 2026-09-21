"""Maximal articulated internal contacts and motors conserve momentum."""

import unittest
from unittest.mock import patch

import numpy as np
import warp as wp

import newton
from newton._src.solvers.phoenx.tests.test_reduced_articulation import _total_momentum


class TestMaximalContactConservation(unittest.TestCase):
    def test_tree_post_integration_contact_momentum(self):
        self._check_self_contact_momentum(steps=1, radius=0.2, at_rest=False)

    def test_direct_post_integration_contact_momentum(self):
        with patch("newton._src.solvers.phoenx.solver.find_full_coordinate_revolute_trees", return_value=[]):
            self._check_self_contact_momentum(steps=1, radius=0.2, at_rest=False)

    def test_tree_sustained_contact_motor_momentum(self):
        distance = np.sqrt(0.2**2 + 0.15**2 + 2 * 0.2 * 0.15 * np.cos(0.3))
        self._check_self_contact_momentum(steps=200, radius=float(0.5 * distance + 5e-6), at_rest=True)

    def test_direct_sustained_contact_motor_momentum(self):
        distance = np.sqrt(0.2**2 + 0.15**2 + 2 * 0.2 * 0.15 * np.cos(0.3))
        with patch("newton._src.solvers.phoenx.solver.find_full_coordinate_revolute_trees", return_value=[]):
            self._check_self_contact_momentum(steps=200, radius=float(0.5 * distance + 5e-6), at_rest=True)

    def _check_self_contact_momentum(self, *, steps, radius, at_rest):
        if not wp.is_cuda_available():
            self.skipTest("CUDA required")
        builder = newton.ModelBuilder(gravity=wp.vec3(0.0))
        root = builder.add_link()
        middle = builder.add_link(mass=1.0, inertia=wp.mat33(0.003, 0.0, 0.0, 0.0, 0.003, 0.0, 0.0, 0.0, 0.003))
        tip = builder.add_link()
        cfg = newton.ModelBuilder.ShapeConfig(mu=0.5, density=1000.0)
        builder.add_shape_sphere(root, radius=radius, cfg=cfg)
        builder.add_shape_sphere(tip, radius=radius, cfg=cfg)
        joints = [builder.add_joint_free(root)]
        joints.append(
            builder.add_joint_revolute(
                root, middle, axis=newton.Axis.Z, parent_xform=wp.transform(wp.vec3(0.2, 0.0, 0.0), wp.quat_identity())
            )
        )
        joints.append(
            builder.add_joint_revolute(
                middle, tip, axis=newton.Axis.Z, parent_xform=wp.transform(wp.vec3(0.15, 0.0, 0.0), wp.quat_identity())
            )
        )
        builder.add_articulation(joints)
        model = builder.finalize(device="cuda:0")
        state = model.state()
        q = state.joint_q.numpy()
        q[-2:] = [0.3, -0.6]
        state.joint_q.assign(q)
        qd = state.joint_qd.numpy()
        if not at_rest:
            qd[:3] = [0.4, -0.15, 0.0]
            qd[5:] = [2.0, 3.0, -2.5]
        state.joint_qd.assign(qd)
        newton.eval_fk(model, state.joint_q, state.joint_qd, state)
        control = model.control()
        joint_force = control.joint_f.numpy()
        joint_force[-2 if at_rest else -1] = 0.1
        control.joint_f.assign(joint_force)
        pipeline = newton.CollisionPipeline(model, rigid_contact_max=32, contact_matching="sticky")
        solver = newton.solvers.SolverPhoenX(
            model,
            collision_pipeline=pipeline,
            articulation_mode="maximal",
            step_layout="single_world",
            substeps=1,
            solver_iterations=8,
            sor_boost=1.0,
        )
        contacts = pipeline.contacts()
        before = _total_momentum(model, state)

        import functools

        def report(label):
            solver._export_body_state(state, 0.001)
            print(label, _total_momentum(model, state) - before, flush=True)

        def wrap(obj, name):
            old = getattr(obj, name)

            @functools.wraps(old)
            def f(*a, **kw):
                report(name + " BEFORE " + str(kw))
                result = old(*a, **kw)
                report(name + " AFTER " + str(kw))
                return result

            setattr(obj, name, f)

        world = solver.world
        for name in [
            "_integrate_forces_and_gravity",
            "_integrate_positions",
            "_solve_direct_contacts",
            "_solve_maximal_articulated_contacts",
        ]:
            wrap(world, name)
        wrap(world._direct_equality_system, "solve")
        pipeline.collide(state, contacts)
        solver.step(state, state, control, contacts, 0.001)
        report("FINAL")


if __name__ == "__main__":
    TestMaximalContactConservation().test_tree_post_integration_contact_momentum()
