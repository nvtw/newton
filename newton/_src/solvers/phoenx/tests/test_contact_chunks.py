# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Contact chunks retain points and physical momentum."""

import unittest

import numpy as np
import warp as wp

import newton
from newton._src.solvers.phoenx.constraints.constraint_contact import (
    ContactColumnContainer,
    contact_column_container_zeros,
    contact_get_body1,
    contact_get_body2,
    contact_get_contact_count,
    contact_get_contact_first,
    contact_set_body1,
    contact_set_body2,
    contact_set_contact_count,
    contact_set_contact_first,
)
from newton._src.solvers.phoenx.constraints.contact_chunks import (
    _count_chunks,
    _publish_chunks,
    _split_columns,
)
from newton._src.solvers.phoenx.tests.test_contact_coupling import _total_momentum


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


@wp.kernel
def _read_ranges(cols: ContactColumnContainer, active: wp.array[wp.int32], ranges: wp.array2d[wp.int32]):
    i = wp.tid()
    if i < active[0]:
        ranges[i, 0] = contact_get_contact_first(cols, i)
        ranges[i, 1] = contact_get_contact_count(cols, i)


@wp.kernel
def _seed_ranges(cols: ContactColumnContainer, ranges: wp.array2d[wp.int32]):
    i = wp.tid()
    contact_set_contact_first(cols, i, ranges[i, 0])
    contact_set_contact_count(cols, i, ranges[i, 1])
    contact_set_body1(cols, i, 2 * i)
    contact_set_body2(cols, i, 2 * i + 1)
    cols.articulation_owner[i] = i


@wp.kernel
def _read_endpoints(cols: ContactColumnContainer, active: wp.array[wp.int32], endpoints: wp.array2d[wp.int32]):
    i = wp.tid()
    if i < active[0]:
        endpoints[i, 0] = contact_get_body1(cols, i)
        endpoints[i, 1] = contact_get_body2(cols, i)


class TestContactChunks(unittest.TestCase):
    def test_multiple_pairs_and_recycled_ranges(self):
        """Retain pair metadata and exact point ownership while columns shrink."""
        device = "cpu"
        capacity = 32
        source = contact_column_container_zeros(capacity, device=device)
        target = contact_column_container_zeros(capacity, device=device)
        source_pair = wp.array([4, 8, 2] + [0] * 29, dtype=wp.int32, device=device)
        target_pair = wp.full(capacity, -1, dtype=wp.int32, device=device)
        active = wp.array([3], dtype=wp.int32, device=device)
        counts = wp.zeros(capacity, dtype=wp.int32, device=device)
        offsets = wp.zeros_like(counts)
        cid = wp.full(capacity, -1, dtype=wp.int32, device=device)
        total_active = wp.zeros(1, dtype=wp.int32, device=device)
        for ranges, expected_starts, expected_counts, expected_pairs in (
            ([[0, 13], [13, 2], [15, 8]], [0, 6, 12, 13, 15, 21], [6, 6, 1, 2, 6, 2], [4, 4, 4, 8, 2, 2]),
            ([[0, 3]], [0], [3], [4]),
        ):
            active.assign([len(ranges)])
            initial = wp.array(ranges, dtype=wp.int32, device=device)
            wp.launch(_seed_ranges, len(ranges), [source, initial], device=device)
            wp.launch(_count_chunks, capacity, [source, active, 6, counts], device=device)
            wp.utils.array_scan(counts, offsets, inclusive=False)
            wp.launch(
                _split_columns,
                capacity,
                [source, target, active, counts, offsets, 6, source_pair, target_pair, 7, cid],
                device=device,
            )
            wp.launch(_publish_chunks, 1, [counts, offsets, active, 7, total_active], device=device)
            actual = wp.zeros((capacity, 2), dtype=wp.int32, device=device)
            wp.launch(_read_ranges, capacity, [target, active, actual], device=device)
            size = len(expected_starts)
            self.assertEqual(int(active.numpy()[0]), size)
            self.assertEqual(int(total_active.numpy()[0]), size + 7)
            np.testing.assert_array_equal(actual.numpy()[:size, 0], expected_starts)
            np.testing.assert_array_equal(actual.numpy()[:size, 1], expected_counts)
            np.testing.assert_array_equal(target_pair.numpy()[:size], expected_pairs)
            endpoint_map = {4: (0, 1), 8: (2, 3), 2: (4, 5)}
            endpoints = wp.zeros((capacity, 2), dtype=wp.int32, device=device)
            wp.launch(_read_endpoints, capacity, [target, active, endpoints], device=device)
            np.testing.assert_array_equal(endpoints.numpy()[:size], [endpoint_map[pair] for pair in expected_pairs])
            np.testing.assert_array_equal(
                target.articulation_owner.numpy()[:size], [endpoint_map[pair][0] // 2 for pair in expected_pairs]
            )

            expected_cid = np.repeat(np.arange(size) + 7, expected_counts)
            np.testing.assert_array_equal(cid.numpy()[: len(expected_cid)], expected_cid)

    def test_disabled_and_empty_contact_scratch(self):
        """Opt-in empty scenes allocate no scratch and accept absent contacts."""
        builder = newton.ModelBuilder(gravity=wp.vec3(0.0))
        builder.add_body(mass=1.0, inertia=wp.mat33(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0))
        model = builder.finalize(device="cpu")
        for chunk_size in (0, 6):
            solver = newton.solvers.SolverPhoenX(model, contact_chunk_size=chunk_size, step_layout="single_world")
            self.assertIsNone(solver.world._contact_chunk_scratch)
            solver.world._ingest_and_warmstart_contacts(None, None)
            self.assertEqual(int(solver.world._num_active_constraints.numpy()[0]), solver.world._contact_offset)
        for invalid in (-1, True, 1.5, "6"):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                newton.solvers.SolverPhoenX(model, contact_chunk_size=invalid)
        with self.assertRaises(ValueError):
            newton.solvers.SolverPhoenX(model, contact_chunk_size=6, contact_friction_model="patch")

    def test_two_world_point_columns_keep_distinct_endpoints(self):
        """Chunk prefix sums retain complete contacts in each world."""
        builder = newton.ModelBuilder(gravity=wp.vec3(0.0))
        cfg = newton.ModelBuilder.ShapeConfig(density=0.0, mu=0.5, gap=0.001)
        for _ in range(2):
            builder.begin_world()
            for x in (0.0, 1.0000001):
                body = builder.add_body(
                    mass=1.0,
                    inertia=wp.mat33(0.2, 0.0, 0.0, 0.0, 0.2, 0.0, 0.0, 0.0, 0.2),
                    xform=wp.transform(wp.vec3(x, 0.0, 0.0), wp.quat_identity()),
                )
                builder.add_shape_box(body, hx=0.5, hy=0.5, hz=0.5, cfg=cfg)
            builder.end_world()
        model = builder.finalize(device="cuda:0")
        state = model.state()
        pipeline = newton.CollisionPipeline(model, rigid_contact_max=64, contact_matching="sticky")
        contacts = pipeline.contacts()
        solver = newton.solvers.SolverPhoenX(
            model,
            collision_pipeline=pipeline,
            contact_chunk_size=1,
            step_layout="multi_world",
            substeps=1,
            solver_iterations=1,
            velocity_iterations=0,
            sor_boost=1.0,
        )
        pipeline.collide(state, contacts)
        solver.step(state, state, model.control(), contacts, 1e-4)
        with wp.ScopedCapture() as capture:
            solver.step(state, state, model.control(), contacts, 1e-4)
        wp.capture_launch(capture.graph)
        world = solver.world
        active = int(world._ingest_scratch.num_contact_columns.numpy()[0])
        self.assertEqual(active, int(contacts.rigid_contact_count.numpy()[0]))
        self.assertGreaterEqual(active, 8)
        endpoints = wp.zeros((world.max_contact_columns, 2), dtype=wp.int32, device=model.device)
        wp.launch(
            _read_endpoints,
            world.max_contact_columns,
            [world._contact_cols, world._ingest_scratch.num_contact_columns, endpoints],
            device=model.device,
        )
        pairs = np.unique(endpoints.numpy()[:active], axis=0)
        self.assertEqual(len(pairs), 2)
        self.assertEqual(len(set(pairs.ravel())), 4)
        np.testing.assert_array_equal(
            world._cid_of_contact_cur.numpy()[:active], np.arange(active) + world._contact_offset
        )
        self.assertTrue(np.isfinite(state.body_q.numpy()).all())

    def test_split_face_contacts_preserve_momentum_and_energy(self):
        """Split every face point into a copy-safe column without losing rows."""
        for reverse in (False, True):
            with self.subTest(reverse=reverse):
                builder = newton.ModelBuilder(gravity=wp.vec3(0.0))
                cfg = newton.ModelBuilder.ShapeConfig(density=0.0, mu=0.5, gap=0.001)
                positions = (1.0000001, 0.0) if reverse else (0.0, 1.0000001)
                for x in positions:
                    body = builder.add_body(
                        mass=1.0,
                        inertia=wp.mat33(0.2, 0.0, 0.0, 0.0, 0.2, 0.0, 0.0, 0.0, 0.2),
                        xform=wp.transform(wp.vec3(x, 0.0, 0.0), wp.quat_identity()),
                    )
                    builder.add_shape_box(body, hx=0.5, hy=0.5, hz=0.5, cfg=cfg)
                    # Force compound-body ingestion, whose warm-start gather
                    # rewrites the original pair-to-column map.
                    builder.add_shape_sphere(
                        body,
                        radius=0.01,
                        xform=wp.transform(wp.vec3(0.0, 10.0, 0.0), wp.quat_identity()),
                        cfg=cfg,
                    )
                model = builder.finalize(device="cuda:0")
                state = model.state()
                velocities = state.body_qd.numpy()
                for i, x in enumerate(positions):
                    velocities[i, :3] = [0.5 if x == 0.0 else -0.5, 0.15 if x == 0.0 else -0.15, 0.0]
                state.body_qd.assign(velocities)
                pipeline = newton.CollisionPipeline(model, rigid_contact_max=64, contact_matching="sticky")
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
                    solver_iterations=1,
                    velocity_iterations=0,
                    sor_boost=1.0,
                    contact_chunk_size=1,
                )
                before = _total_momentum(model, state)
                before_energy = energy(model, state)
                pipeline.collide(state, contacts)
                point_count = int(contacts.rigid_contact_count.numpy()[0])
                self.assertGreaterEqual(point_count, 4)
                solver.step(state, state, model.control(), contacts, 1.0e-4)
                world = solver.world
                count = int(world._ingest_scratch.num_contact_columns.numpy()[0])
                self.assertEqual(count, point_count)
                self.assertTrue(world._enable_body_pair_grouping)
                np.testing.assert_array_equal(
                    world._cid_of_contact_cur.numpy()[:point_count],
                    np.arange(point_count) + world._contact_offset,
                )
                ranges = wp.zeros((world.max_contact_columns, 2), dtype=wp.int32, device=model.device)
                wp.launch(
                    _read_ranges,
                    world.max_contact_columns,
                    [world._contact_cols, world._ingest_scratch.num_contact_columns, ranges],
                    device=model.device,
                )
                actual = ranges.numpy()[:count]
                np.testing.assert_array_equal(actual[:, 0], np.arange(point_count))
                np.testing.assert_array_equal(actual[:, 1], np.ones(point_count, dtype=np.int32))
                self.assertGreaterEqual(int(world._copy_state.count_per_node.numpy().max()), 4)
                np.testing.assert_allclose(_total_momentum(model, state), before, atol=2e-6, rtol=0)
                self.assertLessEqual(energy(model, state), before_energy + 1e-6)

                # Recycle fewer point slots with reversed history identities.
                # Compound gather rewrites owners using unsplit pairs; the
                # canonical chunk post-pass must restore the new owners.
                old_impulses = world._contact_container.impulses.numpy()
                old_impulses.fill(0.0)
                old_impulses[0, :point_count] = np.arange(point_count) + 1.0
                world._contact_container.impulses.assign(old_impulses)
                new_count = point_count - 1
                contacts.rigid_contact_count.assign([new_count])
                match = contacts.rigid_contact_match_index.numpy()
                match[:new_count] = np.arange(point_count - 1, 0, -1)
                contacts.rigid_contact_match_index.assign(match)
                generation = contacts.contact_generation.numpy()
                contacts.contact_generation.assign(generation + 1)
                world._ingest_and_warmstart_contacts(
                    contacts, world._contact_views.shape_body, world._contact_views.shape_type
                )
                mapped = world._contact_views.rigid_contact_match_index.numpy()[:new_count]
                np.testing.assert_array_equal(
                    world._contact_container.impulses.numpy()[0, :new_count],
                    old_impulses[0, mapped],
                )
                np.testing.assert_array_equal(
                    world._cid_of_contact_cur.numpy()[:new_count],
                    np.arange(new_count) + world._contact_offset,
                )
                self.assertEqual(int(world._ingest_scratch.num_contact_columns.numpy()[0]), new_count)


if __name__ == "__main__":
    unittest.main()
