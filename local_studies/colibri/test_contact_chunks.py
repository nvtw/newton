# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Contact chunks retain points and physical momentum."""

import unittest

import numpy as np
import warp as wp

import newton
from local_studies.colibri.contact_chunks import (
    _count_chunks,
    _publish_chunks,
    _split_columns,
    install,
    reserve_capacity,
)
from local_studies.colibri.test_mass_split_bilateral import energy
from newton._src.solvers.phoenx.constraints.constraint_contact import (
    ContactColumnContainer,
    contact_column_container_zeros,
    contact_get_contact_count,
    contact_get_contact_first,
    contact_set_contact_count,
    contact_set_contact_first,
)
from newton._src.solvers.phoenx.tests.test_contact_coupling import _total_momentum


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
            expected_cid = np.repeat(np.arange(size) + 7, expected_counts)
            np.testing.assert_array_equal(cid.numpy()[: len(expected_cid)], expected_cid)

    def test_split_face_contacts_preserve_momentum_and_energy(self):
        """Split every face point into a copy-safe column without losing rows."""
        reserve_capacity()
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
                )
                install(solver, chunk_size=1)
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


if __name__ == "__main__":
    unittest.main()
