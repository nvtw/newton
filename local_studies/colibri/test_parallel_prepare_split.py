# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Compare pointwise preparation against ordered preparation with changing copies."""

import unittest

import numpy as np
import warp as wp

import newton
from local_studies.colibri.contact_chunks import install as install_chunks
from local_studies.colibri.contact_chunks import reserve_capacity
from local_studies.colibri.parallel_prepare_split import install
from local_studies.colibri.test_mass_split_bilateral import energy
from newton._src.solvers.phoenx.tests.test_contact_coupling import _total_momentum


def run_scene(parallel):
    reserve_capacity()
    builder = newton.ModelBuilder(gravity=wp.vec3(0.0))
    cfg = newton.ModelBuilder.ShapeConfig(density=0.0, mu=0.5, gap=0.001)
    for index, x in enumerate((0.0, 1.0000001, -0.6000001)):
        body = builder.add_body(
            xform=wp.transform(wp.vec3(x, 0.0, 0.0)),
            mass=1.0,
            inertia=wp.mat33(0.2, 0.0, 0.0, 0.0, 0.2, 0.0, 0.0, 0.0, 0.2),
        )
        if index < 2:
            builder.add_shape_box(body, hx=0.5, hy=0.5, hz=0.5, cfg=cfg)
            builder.add_shape_sphere(body, radius=0.01, xform=wp.transform(wp.vec3(0.0, 10.0, 0.0)), cfg=cfg)
        else:
            builder.add_shape_sphere(body, radius=0.1, cfg=cfg)
        direction = 0.0 if index == 0 else (-1.0 if index == 1 else 1.0)
        builder.body_qd[body] = (0.3 * direction, 0.1 * direction, 0.0, 0.0, 0.0, 0.0)
    model = builder.finalize(device="cuda:0")
    state = model.state()
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
    install_chunks(solver, chunk_size=3)
    if parallel:
        install(solver)
    world = solver.world
    reference_momentum = _total_momentum(model, state)
    reference_energy = energy(model, state)
    snapshots = []
    for frame in range(8):
        state.clear_forces()
        pipeline.collide(state, contacts)
        count = int(contacts.rigid_contact_count.numpy()[0])
        assert count == 5, count
        if frame % 2:
            contacts.rigid_contact_count.assign([4])
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
            }
        )
    return snapshots


class TestParallelPrepareSplit(unittest.TestCase):
    def test_ragged_columns_and_changing_copies_match_exactly(self):
        baseline = run_scene(False)
        candidate = run_scene(True)
        self.assertGreater(np.max(baseline[0]["copy_counts"]), np.max(baseline[1]["copy_counts"]))
        for frame, (before, after) in enumerate(zip(baseline, candidate, strict=True)):
            for key in before:
                with self.subTest(frame=frame, field=key):
                    np.testing.assert_array_equal(before[key], after[key])


if __name__ == "__main__":
    unittest.main()
