# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Check bounded contact preparation against the original full-width grid."""

import unittest
from unittest.mock import patch

import numpy as np
import warp as wp

import newton
from newton._src.solvers.phoenx.constraints.contact_container import contact_tile_rows_numpy
from newton._src.solvers.phoenx.tests.test_rigid_split_prepare import run_scene


@unittest.skipUnless(wp.is_cuda_available(), "Contact preparation requires CUDA")
class TestContactPrepareGrid(unittest.TestCase):
    def test_contact_levers_preserve_local_precision(self):
        """Preserve local contact lever arms away from the world origin."""
        for device in ("cpu", "cuda:0"):
            for parallel in (False,) if device == "cpu" else (False, True):
                with self.subTest(device=device, parallel=parallel), wp.ScopedDevice(device):
                    builder = newton.ModelBuilder(gravity=wp.vec3(0.0))
                    cfg = builder.ShapeConfig(density=0, gap=0.002, margin=0)
                    for position, mass in (((64.0, 32.0, 0.0), 0.42), ((64.08, 32.04, 0.1998), 0.57)):
                        body = builder.add_body(
                            xform=wp.transform(position, wp.quat_identity()),
                            mass=mass,
                            inertia=wp.mat33(np.eye(3) * mass / 150),
                        )
                        builder.add_shape_box(body, hx=0.1, hy=0.1, hz=0.1, cfg=cfg)
                    model = builder.finalize(device=device)
                    pipeline = newton.CollisionPipeline(model, rigid_contact_max=32, contact_matching="sticky")
                    solver = newton.solvers.SolverPhoenX(
                        model,
                        collision_pipeline=pipeline,
                        solver_iterations=1,
                        velocity_iterations=0,
                        substeps=1,
                        step_layout="single_world",
                        enable_body_pair_grouping=True,
                        parallel_contact_prepare=parallel,
                        mass_splitting=False,
                    )
                    state, contacts = model.state(), pipeline.contacts()
                    pipeline.collide(state, contacts)
                    dispatcher = type(solver.world._dispatcher)
                    original = dispatcher.solve
                    checked = []

                    def record(instance, idt, *, original=original, model=model, checked=checked):
                        original(instance, idt)
                        world = instance._world
                        cc, scratch, bodies = world._contact_container, world._ingest_scratch, world.bodies
                        count = int(world._cc_valid_count.numpy()[0])
                        self.assertGreater(count, 0)
                        shape_body = model.shape_body.numpy() + 1
                        first = shape_body[scratch.sorted_shape0.numpy()[:count]]
                        second = shape_body[scratch.sorted_shape1.numpy()[:count]]
                        position = bodies.position.numpy().astype(np.float64)
                        com = bodies.body_com.numpy().astype(np.float64)
                        # Both boxes have identity orientations and zero margins.
                        offset0 = scratch.sorted_point0.numpy()[:count].astype(np.float64) - com[first]
                        offset1 = scratch.sorted_point1.numpy()[:count].astype(np.float64) - com[second]
                        inverse = bodies.inverse_mass.numpy().astype(np.float64)
                        weight = (inverse[first] / (inverse[first] + inverse[second]))[:, None]
                        delta = position[second] - position[first] + offset1 - offset0
                        expected0 = offset0 + (1 - weight) * delta
                        expected1 = offset1 - weight * delta
                        derived = contact_tile_rows_numpy(cc, "derived")[:, :count].T
                        # At 64 m, a world-position round trip loses micrometers;
                        # these 0.1 m lever arms should retain float32 local precision.
                        np.testing.assert_allclose(derived[:, 9:12], expected0, rtol=0, atol=2e-7)
                        np.testing.assert_allclose(derived[:, 12:15], expected1, rtol=0, atol=2e-7)
                        checked.append(count)

                    with patch.object(dispatcher, "solve", record):
                        solver.step(state, state, model.control(), contacts, 1 / 240)
                    self.assertTrue(checked)

    def test_bounded_prepare_preserves_contact_state(self):
        """Prepare all columns and preserve momentum with fewer blocks and row lanes."""
        launch = wp.launch

        def run(split, chunk, serial):
            dimensions = []

            def record(kernel, *args, **kwargs):
                if "_get_parallel_contact_prepare_kernel" in kernel.func.__qualname__:
                    if serial:
                        # Reference: one lane walks every row of its column.
                        kwargs["dim"] = (kwargs["dim"][0], 1)
                        kwargs["inputs"] = [*kwargs["inputs"][:-1], wp.int32(1)]
                    else:
                        # Force multiple column visits per group in the small fixture.
                        kwargs["dim"] = (1, kwargs["dim"][1])
                        kwargs["inputs"] = [*kwargs["inputs"][:-2], wp.int32(1), kwargs["inputs"][-1]]
                    dimensions.append(kwargs["dim"])
                return launch(kernel, *args, **kwargs)

            with patch.object(wp, "launch", record):
                states = run_scene(True, mass_splitting=split, chunk_size=chunk)
            self.assertTrue(dimensions)
            self.assertTrue(all(dim[1] == (1 if serial else 8) for dim in dimensions))
            self.assertGreater(int(states[0]["column_count"][0]), 1)
            return states

        for split in (False, True):
            # Chunks wider than the eight row lanes make one lane prepare several rows.
            for chunk in (0, 1, 3, 6, 128, 129):
                with self.subTest(split=split, chunk=chunk):
                    before = run(split, chunk, True)
                    after = run(split, chunk, False)
                    for reference, actual in zip(before, after, strict=True):
                        for name in reference:
                            np.testing.assert_array_equal(
                                reference[name].view(np.uint8), actual[name].view(np.uint8), err_msg=name
                            )


if __name__ == "__main__":
    unittest.main()
