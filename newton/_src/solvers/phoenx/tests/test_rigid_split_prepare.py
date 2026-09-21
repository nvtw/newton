# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Check public parallel preparation against the standard dispatch."""

import unittest

import numpy as np
import warp as wp

import newton
from newton._src.solvers.phoenx.constraints.constraint_contact_cloth import _make_contact_prepare_for_iteration_at
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


def run_scene(parallel, mass_splitting=True, *, chunk_size=0):
    builder = newton.ModelBuilder(gravity=wp.vec3(0.0))
    cfg = newton.ModelBuilder.ShapeConfig(density=0.0, mu=0.5, gap=0.001)
    positions = (
        (0.0, 0.0, 0.0),
        (1.0000001, 0.0, 0.0),
        (-0.6000001, 0.0, 0.0),
        (0.0, 0.6000001, 0.0),
        (0.0, -0.6000001, 0.0),
    )
    for index, position in enumerate(positions):
        body = builder.add_body(
            xform=wp.transform(wp.vec3(position)),
            mass=1.0,
            inertia=wp.mat33(0.2, 0.0, 0.0, 0.0, 0.2, 0.0, 0.0, 0.0, 0.2),
        )
        if index < 2:
            builder.add_shape_box(body, hx=0.5, hy=0.5, hz=0.5, cfg=cfg)
            builder.add_shape_sphere(body, radius=0.01, xform=wp.transform(wp.vec3(0.0, 10.0, 0.0)), cfg=cfg)
        else:
            builder.add_shape_sphere(body, radius=0.1, cfg=cfg)
        direction = -np.asarray(position)
        builder.body_qd[body] = (*tuple(0.3 * direction), 0.0, 0.0, 0.0)
    model = builder.finalize(device="cuda:0")
    state = model.state()
    pipeline = newton.CollisionPipeline(model, rigid_contact_max=64, contact_matching="sticky")
    contacts = pipeline.contacts()
    solver = newton.solvers.SolverPhoenX(
        model,
        collision_pipeline=pipeline,
        joint_mode="maximal_direct",
        step_layout="single_world",
        mass_splitting=mass_splitting,
        parallel_contact_prepare=parallel,
        contact_chunk_size=chunk_size,
        max_colored_partitions=1,
        mass_splitting_batch_size=1,
        substeps=1,
        solver_iterations=1,
        velocity_iterations=0,
        sor_boost=1.0,
    )
    world = solver.world
    reference_momentum = _total_momentum(model, state)
    reference_energy = energy(model, state)
    snapshots = []
    for frame in range(8):
        state.clear_forces()
        pipeline.collide(state, contacts)
        count = int(contacts.rigid_contact_count.numpy()[0])
        assert count == 7, count
        if frame % 2:
            contacts.rigid_contact_count.assign([6])
        solver.step(state, state, model.control(), contacts, 1.0e-4)
        np.testing.assert_allclose(_total_momentum(model, state), reference_momentum, rtol=0.0, atol=2.0e-6)
        assert energy(model, state) <= reference_energy + 1.0e-6
        snapshots.append(
            {
                "q": state.body_q.numpy(),
                "qd": state.body_qd.numpy(),
                "derived": world._contact_container.derived.numpy(),
                "impulses": world._contact_container.impulses.numpy(),
                "anchors": world._contact_container.lambdas.numpy(),
                "copy_velocity": world._copy_state.velocity.numpy(),
                "copy_angular_velocity": world._copy_state.angular_velocity.numpy(),
                "copy_counts": world._copy_state.count_per_node.numpy(),
                "color_starts": world._partitioner.color_starts.numpy(),
                "column_count": world._ingest_scratch.num_contact_columns.numpy(),
                "columns": world._contact_cols.data.numpy(),
                "inverse_mass": world.bodies.inverse_mass.numpy(),
                "inverse_inertia": world.bodies.inverse_inertia_world.numpy(),
            }
        )
    return snapshots


def check_mass_against_fp64(snapshot):
    columns = snapshot["columns"].view(np.int32)
    mass = snapshot["inverse_mass"].astype(np.float64)
    sym = snapshot["inverse_inertia"].astype(np.float64)
    inertia = sym[:, [0, 3, 4, 3, 1, 5, 4, 5, 2]].reshape(-1, 3, 3)
    geometry = snapshot["derived"]
    axes = snapshot["anchors"]
    for column in range(int(snapshot["column_count"][0])):
        body0, body1 = columns[1:3, column]
        first, count = columns[5:7, column]
        copies0, copies1 = columns[9:11, column]
        for point in range(first, first + count):
            normal = axes[:3, point].astype(np.float64)
            tangent = axes[3:6, point].astype(np.float64)
            r0 = geometry[9:12, point].astype(np.float64)
            r1 = geometry[12:15, point].astype(np.float64)
            for row, axis in enumerate((normal, tangent, np.cross(normal, tangent))):
                angular0 = np.cross(r0, axis)
                angular1 = np.cross(r1, axis)
                mobility = copies0 * (mass[body0] + angular0 @ inertia[body0] @ angular0) + copies1 * (
                    mass[body1] + angular1 @ inertia[body1] @ angular1
                )
                expected = 1.0 / mobility if mobility > 1.0e-12 else 0.0
                np.testing.assert_allclose(geometry[row, point], expected, rtol=16 * np.finfo(np.float32).eps, atol=0.0)


class TestParallelPrepareSplit(unittest.TestCase):
    def test_pointwise_geometry_rejects_unsupported_modes(self):
        for option in ("cloth_support", "patch_friction", "packed_rows", "stage_body_properties"):
            with self.subTest(option=option), self.assertRaises(ValueError):
                _make_contact_prepare_for_iteration_at(
                    pointwise_geometry=True, has_mass_splitting=True, **{"cloth_support": False, option: True}
                )

    def test_unsplit_constructor_path_matches(self):
        baseline = run_scene(False, mass_splitting=False)
        candidate = run_scene(True, mass_splitting=False)
        for before, after in zip(baseline, candidate, strict=True):
            check_mass_against_fp64(before)
            check_mass_against_fp64(after)
            for key in before:
                np.testing.assert_array_equal(before[key].view(np.uint8), after[key].view(np.uint8))

    def test_public_constructor_rejects_unsupported_modes(self):
        builder = newton.ModelBuilder()
        builder.add_body(mass=1.0, inertia=wp.mat33(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0))
        model = builder.finalize(device="cuda:0")
        for options in (
            {"step_layout": "multi_world"},
            {"step_layout": "single_world", "contact_friction_model": "patch"},
            {"step_layout": "single_world", "joint_mode": "reduced"},
            {"step_layout": "single_world", "sleeping_velocity_threshold": 0.01},
            {"step_layout": "single_world", "mass_splitting_unrolled": True},
        ):
            with self.subTest(options=options), self.assertRaises(ValueError):
                newton.solvers.SolverPhoenX(model, parallel_contact_prepare=True, **options)

    def test_ragged_columns_and_changing_copies_match_exactly(self):
        baseline = run_scene(False)
        candidate = run_scene(True)
        self.assertGreater(np.max(baseline[0]["copy_counts"]), np.max(baseline[1]["copy_counts"]))
        self.assertGreater(baseline[0]["color_starts"][1], 0)
        self.assertGreater(baseline[0]["color_starts"][2], baseline[0]["color_starts"][1])
        for frame, (before, after) in enumerate(zip(baseline, candidate, strict=True)):
            check_mass_against_fp64(before)
            check_mass_against_fp64(after)
            for key in before:
                with self.subTest(frame=frame, field=key):
                    np.testing.assert_array_equal(before[key].view(np.uint8), after[key].view(np.uint8))


if __name__ == "__main__":
    unittest.main()
