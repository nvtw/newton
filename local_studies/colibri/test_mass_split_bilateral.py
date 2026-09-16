"""Physical invariants of opt-in mass-split bilateral joint blocks."""

import unittest

import numpy as np
import warp as wp

import newton
from local_studies.colibri.bilateral_pgs import install_fused
from newton._src.solvers.phoenx.tests.test_reduced_articulation import _total_momentum


def energy(model, state):
    mass = model.body_mass.numpy()
    inertia = model.body_inertia.numpy()
    q = state.body_q.numpy()
    v = state.body_qd.numpy()
    total = 0.0
    for i in range(model.body_count):
        rotation = np.array(wp.quat_to_matrix(wp.quat(q[i, 3:]))).reshape(3, 3)
        total += 0.5 * mass[i] * np.dot(v[i, :3], v[i, :3])
        total += 0.5 * v[i, 3:] @ (rotation @ inertia[i] @ rotation.T) @ v[i, 3:]
    return float(total)


class TestMassSplitBilateral(unittest.TestCase):
    def test_unequal_copy_counts_preserve_physical_momentum(self):
        """A three-branch star must conserve momentum with unequal copy counts."""
        for reverse in (False, True):
            with self.subTest(reverse=reverse):
                builder = newton.ModelBuilder(gravity=wp.vec3(0.0))
                order = [3, 2, 1, 0] if reverse else [0, 1, 2, 3]
                ids = {}
                masses = [2.0, 1.0, 3.0, 4.0]
                for label in order:
                    ids[label] = builder.add_link(
                        mass=masses[label], inertia=wp.mat33(0.03, 0.0, 0.0, 0.0, 0.04, 0.0, 0.0, 0.0, 0.05)
                    )
                joints = [builder.add_joint_free(ids[0])]
                for label in (1, 2, 3):
                    joints.append(
                        builder.add_joint_fixed(
                            ids[0],
                            ids[label],
                            parent_xform=wp.transform(wp.vec3(0.2 * label, 0.1 * (label % 2), 0.0), wp.quat_identity()),
                        )
                    )
                builder.add_articulation(joints)
                model = builder.finalize(device="cuda:0")
                state = model.state()
                newton.eval_fk(model, state.joint_q, state.joint_qd, state)
                velocities = state.body_qd.numpy()
                for label, body in ids.items():
                    velocities[body] = [0.3 * label - 0.4, 0.2 - 0.1 * label, 0.15 * label, 0.0, 0.0, 0.0]
                state.body_qd.assign(velocities)
                solver = newton.solvers.SolverPhoenX(
                    model,
                    articulation_mode="maximal",
                    step_layout="single_world",
                    mass_splitting=True,
                    max_colored_partitions=0,
                    mass_splitting_batch_size=1,
                    substeps=1,
                    solver_iterations=1,
                    sor_boost=1.0,
                )
                install_fused(solver, mass_splitting=True)
                before = _total_momentum(model, state)
                before_energy = energy(model, state)
                solver.step(state, state, model.control(), None, 0.0001)
                counts = solver.world._copy_state.count_per_node.numpy()
                self.assertGreaterEqual(int(counts[ids[0] + 1]), 3)
                self.assertEqual(int(counts[ids[1] + 1]), 1)
                np.testing.assert_allclose(_total_momentum(model, state), before, rtol=0, atol=2e-5)
                self.assertLessEqual(energy(model, state), before_energy + 1e-6)

    def test_contact_count_changes_preserve_momentum(self):
        """Changing balanced internal contact rows must not leak momentum."""
        builder = newton.ModelBuilder(gravity=wp.vec3(0.0))
        root = builder.add_link()
        middle = builder.add_link(mass=1.0, inertia=wp.mat33(0.003, 0.0, 0.0, 0.0, 0.003, 0.0, 0.0, 0.0, 0.003))
        tip = builder.add_link()
        cfg = newton.ModelBuilder.ShapeConfig(mu=0.5, density=1000.0)
        builder.add_shape_sphere(root, radius=0.2, cfg=cfg)
        builder.add_shape_sphere(tip, radius=0.2, cfg=cfg)
        joints = [builder.add_joint_free(root)]
        joints.append(
            builder.add_joint_revolute(
                root, middle, axis=newton.Axis.Z, parent_xform=wp.transform(wp.vec3(0.2, 0, 0), wp.quat_identity())
            )
        )
        joints.append(
            builder.add_joint_revolute(
                middle, tip, axis=newton.Axis.Z, parent_xform=wp.transform(wp.vec3(0.15, 0, 0), wp.quat_identity())
            )
        )
        builder.add_articulation(joints)
        model = builder.finalize(device="cuda:0")
        state = model.state()
        q = state.joint_q.numpy()
        q[-2:] = [0.3, -0.6]
        state.joint_q.assign(q)
        qd = state.joint_qd.numpy()
        qd[:] = [0.4, -0.15, 0.0, 0.0, 0.0, 0.2, 0.3, -0.25]
        state.joint_qd.assign(qd)
        newton.eval_fk(model, state.joint_q, state.joint_qd, state)
        pipeline = newton.CollisionPipeline(model, rigid_contact_max=32, contact_matching="sticky")
        contacts = pipeline.contacts()
        solver = newton.solvers.SolverPhoenX(
            model,
            collision_pipeline=pipeline,
            articulation_mode="maximal",
            step_layout="single_world",
            mass_splitting=True,
            max_colored_partitions=0,
            mass_splitting_batch_size=1,
            substeps=1,
            solver_iterations=2,
            sor_boost=1.0,
            prepare_refresh_stride=1,
        )
        install_fused(solver, mass_splitting=True)
        before = _total_momentum(model, state)
        root_counts = []
        for step in range(12):
            state.clear_forces()
            pipeline.collide(state, contacts)
            # Exercise graph/history recycling. Each enabled contact still
            # applies balanced internal impulses; disabled rows apply none.
            if step % 2:
                contacts.rigid_contact_count.zero_()
            else:
                self.assertGreater(int(contacts.rigid_contact_count.numpy()[0]), 0)
            solver.step(state, state, model.control(), contacts, 0.0001)
            root_counts.append(int(solver.world._copy_state.count_per_node.numpy()[root + 1]))
            np.testing.assert_allclose(_total_momentum(model, state), before, rtol=0, atol=2e-4)
        self.assertEqual(set(root_counts), {1, 2})

    def test_bounded_motor_with_unequal_copy_counts(self):
        """Copy scaling must retain the physical drive impulse bound."""
        builder = newton.ModelBuilder(gravity=wp.vec3(0.0))
        links = [
            builder.add_link(mass=m, inertia=wp.mat33(0.03, 0.0, 0.0, 0.0, 0.04, 0.0, 0.0, 0.0, 0.05))
            for m in (2.0, 1.0, 3.0, 4.0)
        ]
        joints = [builder.add_joint_free(links[0])]
        joints.append(
            builder.add_joint_revolute(
                links[0],
                links[1],
                axis=newton.Axis.Z,
                parent_xform=wp.transform(wp.vec3(0.2, 0, 0), wp.quat_identity()),
                target_vel=4.0,
                target_kd=10.0,
                effort_limit=0.1,
            )
        )
        for i in (2, 3):
            joints.append(
                builder.add_joint_fixed(
                    links[0], links[i], parent_xform=wp.transform(wp.vec3(0.2 * i, 0, 0), wp.quat_identity())
                )
            )
        builder.add_articulation(joints)
        model = builder.finalize(device="cuda:0")
        state = model.state()
        newton.eval_fk(model, state.joint_q, state.joint_qd, state)
        solver = newton.solvers.SolverPhoenX(
            model,
            articulation_mode="maximal",
            step_layout="single_world",
            mass_splitting=True,
            max_colored_partitions=0,
            mass_splitting_batch_size=1,
            substeps=1,
            solver_iterations=8,
            sor_boost=1.0,
            prepare_refresh_stride=1,
        )
        install_fused(solver, mass_splitting=True)
        initial = _total_momentum(model, state)
        dt = 0.0001
        peak = 0.0
        for _ in range(4):
            state.clear_forces()
            solver.step(state, state, model.control(), None, dt)
            direct = solver._direct_equality_system
            impulse = direct.accumulated_impulse.numpy()[direct.row_dynamic.numpy()]
            self.assertEqual(len(impulse), 1)
            peak = max(peak, float(np.max(np.abs(impulse))))
            self.assertLessEqual(float(np.max(np.abs(impulse))), 0.1 * dt + 1e-8)
            np.testing.assert_allclose(_total_momentum(model, state), initial, rtol=0, atol=2e-5)
        counts = solver.world._copy_state.count_per_node.numpy()
        self.assertEqual(int(counts[links[0] + 1]), 3)
        self.assertEqual(int(counts[links[1] + 1]), 1)
        self.assertGreater(peak, 0.5 * 0.1 * dt)


if __name__ == "__main__":
    unittest.main()
