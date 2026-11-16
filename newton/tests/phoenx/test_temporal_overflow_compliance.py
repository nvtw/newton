# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Keep physical normal compliance independent of numerical body copies."""

import math
import unittest

import numpy as np
import warp as wp

import newton
from newton.solvers import SolverBase, SolverPhoenX


@wp.struct
class _ActiveShapes:
    enabled: wp.array[wp.int32]


@wp.func
def _filter_active_shapes(pair: wp.vec2i, data: _ActiveShapes) -> wp.int32:
    return data.enabled[pair[0]] * data.enabled[pair[1]]


@unittest.skipUnless(wp.is_cuda_available(), "Temporal overflow requires CUDA")
class TestTemporalOverflowCompliance(unittest.TestCase):
    def _check_equilibrium(self, *, authored):
        with wp.ScopedDevice("cuda:0"):
            builder = newton.ModelBuilder()
            body = builder.add_body(xform=wp.transform((0, 0, 0.01)), mass=1.0, inertia=wp.mat33(np.eye(3)))
            cfg = builder.ShapeConfig(density=0, mu=0, mu_static=0, gap=0.002, margin=0)
            # Coincident supports isolate sixteen independent normal springs
            # without rocking. Their physical stiffnesses add in parallel.
            feet = [builder.add_shape_sphere(body, radius=0.01, cfg=cfg) for _ in range(16)]
            ground = builder.add_ground_plane(cfg=cfg)
            model = builder.finalize()
            state, control = model.state(), model.control()
            pipeline = newton.CollisionPipeline(
                model,
                broad_phase="nxn",
                contact_matching="sticky",
                rigid_contact_max=64,
                broad_phase_filter=(_filter_active_shapes, _ActiveShapes),
            )
            filter_data = _ActiveShapes()
            filter_data.enabled = wp.ones(model.shape_count, dtype=wp.int32)
            pipeline.set_broad_phase_filter_data(filter_data)
            contacts = newton.Contacts(
                64, 0, device=model.device, per_contact_shape_properties=authored, contact_matching=True
            )
            solver = SolverPhoenX(
                model,
                collision_pipeline=pipeline,
                substeps=1,
                solver_iterations=24,
                temporal_iterations=True,
                velocity_iterations=1,
                step_layout="single_world",
                contact_hertz=80,
                sleeping_velocity_threshold=0,
                mass_splitting=True,
                mass_splitting_overflow_only=True,
                mass_splitting_unrolled=True,
                mass_splitting_batch_size=1,
                max_colored_partitions=0,
                enable_body_pair_grouping=False,
            )
            observable = solver.observables({SolverBase.ObservableKind.CONTACT_F}, contacts=contacts)
            with wp.ScopedCapture() as capture:
                pipeline.collide(state, contacts, dt=1 / 120)
                if authored:
                    contacts.rigid_contact_stiffness.fill_(400.0)
                    contacts.rigid_contact_damping.fill_(40.0)
                solver.step(state, state, control, contacts, 1 / 120, observables=observable)
            for active in (16, 8, 16):
                with self.subTest(authored=authored, active=active):
                    enabled = np.ones(model.shape_count, dtype=np.int32)
                    enabled[feet[active:]] = 0
                    filter_data.enabled.assign(enabled)
                    penetration, support = [], []
                    for frame in range(240):
                        wp.capture_launch(capture.graph)
                        if frame >= 180:
                            count = int(contacts.rigid_contact_count.numpy()[0])
                            self.assertEqual(count, active)
                            penetration.append(0.01 - float(state.body_q.numpy()[body, 2]))
                            shape0 = contacts.rigid_contact_shape0.numpy()[:count]
                            shape1 = contacts.rigid_contact_shape1.numpy()[:count]
                            force_z = observable.contact_f.numpy()[:count, 2]
                            on_body = np.concatenate((force_z[shape1 == ground], -force_z[shape0 == ground]))
                            self.assertTrue(np.all(on_body >= -1e-6))
                            support.append(float(on_body.sum()))
                    copies = int(solver.world._copy_state.count_per_node.numpy()[body + 1])
                    self.assertEqual(copies, active)
                    stiffness = 400.0 if authored else (2 * math.pi * 80) ** 2
                    expected = 9.81 / (active * stiffness)
                    self.assertAlmostEqual(float(np.mean(penetration)) / expected, 1.0, delta=0.01)
                    self.assertAlmostEqual(float(np.mean(support)), 9.81, delta=0.001)
                    self.assertLess(float(np.linalg.norm(state.body_qd.numpy()[body, :3])), 1e-5)

    def test_frequency_spring_survives_regrouping(self):
        """Preserve analytic spring compliance as collision filtering changes copy counts."""
        self._check_equilibrium(authored=False)

    def test_authored_stiffness_survives_regrouping(self):
        """Retain authored absolute stiffness and damping with changing body copies."""
        self._check_equilibrium(authored=True)


if __name__ == "__main__":
    unittest.main()
