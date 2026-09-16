"""Native physical-head and overflow impulse accounting fixture."""

import importlib
import os
import unittest
from unittest.mock import patch

import numpy as np
import warp as wp

import newton
from newton._src.solvers.phoenx.tests.test_color_group_conservation import _physical_totals


class TestPhysicalHeadPhysics(unittest.TestCase):
    def test_head_overflow_warm_and_joint_momentum(self):
        """Stationary supports account for all momentum changes and add no relax energy."""
        self._run_fixture(False)

    def test_overflow_joint_counts_and_warm_momentum(self):
        """Overflow factors match unequal copy counts and preserve warm impulse momentum."""
        self._run_fixture(True)

    def _run_fixture(self, overflow_joint):
        builder = newton.ModelBuilder(gravity=(0, 0, 0))
        links = []
        for x, mass in ((-0.6, 1.0), (0.6, 3.0)):
            body = builder.add_link(
                xform=wp.transform(wp.vec3(x, 0, 0.5), wp.quat_identity()),
                mass=mass,
                inertia=wp.mat33(mass * 0.2, 0, 0, 0, mass * 0.3, 0, 0, 0, mass * 0.4),
            )
            cfg = newton.ModelBuilder.ShapeConfig(density=0, mu=0.5, gap=0.001)
            if mass == 1.0:
                builder.add_shape_box(body, hx=0.5, hy=0.5, hz=0.5, cfg=cfg)
            else:
                builder.add_shape_sphere(body, radius=0.5, cfg=cfg)
            links.append(body)
        hinge = builder.add_joint_revolute(
            parent=links[0],
            child=links[1],
            axis=(0, 0, 1),
            parent_xform=wp.transform(wp.vec3(0.6, 0, 0), wp.quat_identity()),
            child_xform=wp.transform(wp.vec3(-0.6, 0, 0), wp.quat_identity()),
        )
        joints = [hinge]
        if overflow_joint:
            body = builder.add_link(
                xform=wp.transform(wp.vec3(-0.6, 1.0, 0.5), wp.quat_identity()),
                mass=2.0,
                inertia=wp.mat33(0.4, 0, 0, 0, 0.6, 0, 0, 0, 0.8),
            )
            joints.append(
                builder.add_joint_revolute(
                    parent=links[0],
                    child=body,
                    axis=(0, 0, 1),
                    parent_xform=wp.transform(wp.vec3(0, 0.5, 0), wp.quat_identity()),
                    child_xform=wp.transform(wp.vec3(0, -0.5, 0), wp.quat_identity()),
                )
            )
        builder.add_articulation(joints)
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
            mass_splitting_color_group_size=0,
            max_colored_partitions=1,
            mass_splitting_batch_size=1,
            contact_chunk_size=1,
            parallel_contact_prepare=True,
            substeps=1,
            solver_iterations=2,
            velocity_iterations=1,
            sor_boost=1.0,
        )
        prototype = importlib.import_module(
            os.environ.get("PHYSICAL_HEAD_PROTOTYPE", "local_studies.colibri.physical_head_overflow")
        )
        prototype.configure_world(solver.world)
        world = solver.world
        state = model.state()
        qd = state.body_qd.numpy()
        qd[:2, :3] = [[0.2, 0.1, -0.1], [0.1, -0.1, -0.1]]
        qd[:2, 3:] = [[0.05, 0.1, 0.2], [-0.05, 0.1, -0.1]]
        state.body_qd.assign(qd)
        records = []
        warm_seen = []
        off_center_seen = []
        unequal_seen = []
        overflow_warm_seen = []
        dispatcher = type(world._dispatcher)
        solve, relax = dispatcher.solve, dispatcher.relax

        def observed(original, phase):
            def call(dispatcher, idt):
                before, energy_before = _physical_totals(world)
                old = world._contact_container.impulses.numpy().copy()
                if phase == "biased":
                    warm_seen.append(bool(np.any(old != 0.0)))
                    old.fill(0.0)
                if overflow_joint:
                    ids = world._partitioner.element_ids_by_color.numpy()
                    starts = world._partitioner.color_starts.numpy()
                    self.assertIn(1, ids[starts[1] : starts[2]], "Second joint must be in overflow")
                    expected = world._copy_state.count_per_node.numpy()[[1, 3]]
                    self.assertGreater(max(expected), 1)
                    self.assertNotEqual(expected[0], expected[1])
                    np.testing.assert_array_equal(
                        world.constraints.count_cache.numpy()[1, :2],
                        expected,
                        err_msg="Overflow joint factor must use native iteration copy counts",
                    )
                    data = world.constraints.bilateral
                    rows = data.row_indices.numpy()[1, : data.row_count.numpy()[1]]
                    overflow_warm_seen.append(bool(np.any(data.accumulated.numpy()[rows] != 0)))
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
                        off_center_seen.append(
                            float(np.linalg.norm(point - positions[a if inverse_mass[a] > 0 else b]))
                        )
                        external[:3] += impulse
                        external[3:] += np.cross(point, impulse)
                np.testing.assert_allclose(after - before, external, atol=2e-6, rtol=0)
                if phase == "relax":
                    self.assertLessEqual(energy_after, energy_before + 2e-6)
                counts = world._copy_state.count_per_node.numpy()
                unequal_seen.append(bool(counts[1] != counts[2] and max(counts[1:3]) > 1))
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
        if overflow_joint:
            self.assertTrue(any(overflow_warm_seen), "Overflow joint never retained nonzero impulses")
        self.assertIn("biased", records)
        self.assertIn("relax", records)
        self.assertTrue(any(warm_seen), "Fixture never exercised persistent contact warm impulses")
        self.assertTrue(any(unequal_seen), "Fixture did not produce unequal overflow counts")
        self.assertGreater(max(off_center_seen), 0.1, "Fixture requires off-center support wrenches")


if __name__ == "__main__":
    unittest.main()
