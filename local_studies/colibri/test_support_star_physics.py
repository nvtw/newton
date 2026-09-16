"""Independent external wrench balance for grouped supported bodies."""

import unittest
from unittest.mock import patch

import numpy as np
import warp as wp

import newton
from local_studies.colibri.support_star_groups import attach
from newton._src.solvers.phoenx.tests.test_color_group_conservation import _physical_totals


class TestSupportStarPhysics(unittest.TestCase):
    def test_support_reaction_and_internal_joint_momentum(self):
        """Stationary supports account for all momentum changes and add no relax energy."""
        builder = newton.ModelBuilder(gravity=(0, 0, 0))
        links = []
        for x, mass in ((-0.6, 1.0), (0.6, 3.0)):
            body = builder.add_link(
                xform=wp.transform(wp.vec3(x, 0, 0.5), wp.quat_identity()),
                mass=mass,
                inertia=wp.mat33(mass * 0.2, 0, 0, 0, mass * 0.3, 0, 0, 0, mass * 0.4),
            )
            builder.add_shape_box(
                body, hx=0.5, hy=0.5, hz=0.5, cfg=newton.ModelBuilder.ShapeConfig(density=0, mu=0.5, gap=0.001)
            )
            links.append(body)
        hinge = builder.add_joint_revolute(
            parent=links[0],
            child=links[1],
            axis=(0, 0, 1),
            parent_xform=wp.transform(wp.vec3(0.6, 0, 0), wp.quat_identity()),
            child_xform=wp.transform(wp.vec3(-0.6, 0, 0), wp.quat_identity()),
        )
        builder.add_articulation([hinge])
        builder.add_ground_plane()
        model = builder.finalize(device="cuda:0")
        pipeline = newton.CollisionPipeline(model, rigid_contact_max=128, contact_matching="sticky")
        contacts = pipeline.contacts()
        solver = newton.solvers.SolverPhoenX(
            model,
            collision_pipeline=pipeline,
            articulation_mode="maximal",
            joint_solver="block_pgs",
            step_layout="single_world",
            mass_splitting=True,
            mass_splitting_color_group_size=4,
            contact_chunk_size=1,
            parallel_contact_prepare=True,
            substeps=1,
            solver_iterations=2,
            velocity_iterations=1,
            sor_boost=1.0,
        )
        attach(solver.world)
        world = solver.world
        state = model.state()
        qd = state.body_qd.numpy()
        qd[:, :3] = [[0.2, 0.1, -0.1], [0.1, -0.1, -0.1]]
        qd[:, 3:] = [[0.05, 0.1, 0.2], [-0.05, 0.1, -0.1]]
        state.body_qd.assign(qd)
        records = []
        dispatcher = type(world._dispatcher)
        solve, relax = dispatcher.solve, dispatcher.relax

        def observed(original, phase):
            def call(dispatcher, idt):
                before, energy_before = _physical_totals(world)
                old = world._contact_container.impulses.numpy().copy()
                if phase == "biased":
                    old.fill(0.0)
                original(dispatcher, idt)
                after, energy_after = _physical_totals(world)
                new = world._contact_container.impulses.numpy()
                cc = world._contact_container.lambdas.numpy().astype(np.float64)
                derived = world._contact_container.derived.numpy().astype(np.float64)
                h = world._contact_cols.data.numpy().view(np.int32)
                positions = world.bodies.position.numpy().astype(np.float64)
                inverse_mass = world.bodies.inverse_mass.numpy()
                external = np.zeros(6)
                for column in range(int(world._ingest_scratch.num_contact_columns.numpy()[0])):
                    a, b = h[1:3, column]
                    if inverse_mass[a] > 0 and inverse_mass[b] > 0:
                        continue
                    first, count = h[5:7, column]
                    for k in range(first, first + count):
                        n, t = cc[:3, k], cc[3:6, k]
                        impulse = (
                            (new[0, k] - old[0, k]) * n
                            + (new[1, k] - old[1, k]) * t
                            + (new[2, k] - old[2, k]) * np.cross(n, t)
                        )
                        if inverse_mass[a] > 0:
                            impulse = -impulse
                            point = positions[a] + derived[9:12, k]
                        else:
                            point = positions[b] + derived[12:15, k]
                        external[:3] += impulse
                        external[3:] += np.cross(point, impulse)
                np.testing.assert_allclose(after - before, external, atol=2e-6, rtol=0)
                if phase == "relax":
                    self.assertLessEqual(energy_after, energy_before + 2e-6)
                counts = world._copy_state.count_per_node.numpy()
                self.assertEqual(counts[1], 1)
                self.assertEqual(counts[2], 1)
                records.append(phase)

            return call

        with (
            patch.object(dispatcher, "solve", observed(solve, "biased")),
            patch.object(dispatcher, "relax", observed(relax, "relax")),
        ):
            for _ in range(4):
                state.clear_forces()
                pipeline.collide(state, contacts)
                solver.step(state, state, model.control(), contacts, 0.001)
        self.assertIn("biased", records)
        self.assertIn("relax", records)


if __name__ == "__main__":
    unittest.main()
