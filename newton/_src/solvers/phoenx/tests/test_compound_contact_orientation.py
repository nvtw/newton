# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Compound contact rows must share the column's endpoint convention."""

import unittest

import numpy as np
import warp as wp

import newton
from newton._src.solvers.phoenx.tests.test_reduced_articulation import _total_momentum


@wp.kernel
def _reverse_alternate_contacts(
    count: wp.array[int],
    shape0: wp.array[int],
    shape1: wp.array[int],
    point0: wp.array[wp.vec3],
    point1: wp.array[wp.vec3],
    margin0: wp.array[float],
    margin1: wp.array[float],
    normal: wp.array[wp.vec3],
):
    k = wp.tid()
    if k < count[0] and k % 2 == 1:
        a = wp.int32(shape0[k])
        b = wp.int32(shape1[k])
        p = wp.vec3(point0[k])
        q = wp.vec3(point1[k])
        m = wp.float32(margin0[k])
        n = wp.float32(margin1[k])
        shape0[k] = b
        shape1[k] = a
        point0[k] = q
        point1[k] = p
        margin0[k] = n
        margin1[k] = m
        normal[k] = -normal[k]


@unittest.skipUnless(wp.is_cuda_available(), "PhoenX contact solve requires CUDA")
class TestCompoundContactOrientation(unittest.TestCase):
    def test_soft_reversed_rows(self):
        self._check_reversed_rows("soft")

    def test_tgs_reversed_rows(self):
        self._check_reversed_rows("tgs")

    def _check_reversed_rows(self, scheme):
        """Keep dynamic compound contact results invariant under endpoint reversal."""
        builder = newton.ModelBuilder(gravity=wp.vec3(0.0))
        cfg = builder.ShapeConfig(density=0.0, mu=0.5, gap=0.001)
        for x in (-0.500001, 0.500001):
            body = builder.add_body(
                xform=wp.transform(wp.vec3(x, 0.0, 0.0), wp.quat_identity()),
                mass=1.0,
                inertia=wp.mat33(0.2, 0.0, 0.0, 0.0, 0.3, 0.0, 0.0, 0.0, 0.4),
            )
            for y in (-0.3, 0.3):
                builder.add_shape_box(
                    body,
                    hx=0.5,
                    hy=0.2,
                    hz=0.2,
                    xform=wp.transform(wp.vec3(0.0, y, 0.0), wp.quat_identity()),
                    cfg=cfg,
                )
        model = builder.finalize(device="cuda:0")
        model.request_contact_attributes("force")
        results = []
        for reverse in (False, True):
            state = model.state()
            state.body_qd.assign(np.array([[1.0, 0.2, 0, 0, 0, 0.1], [-1.0, -0.1, 0, 0, 0, -0.1]], dtype=np.float32))
            initial_momentum = _total_momentum(model, state)
            pipeline = newton.CollisionPipeline(model, rigid_contact_max=64, contact_matching="sticky")
            contacts = pipeline.contacts()
            solver = newton.solvers.SolverPhoenX(
                model,
                collision_pipeline=pipeline,
                step_layout="single_world",
                joint_mode="maximal_pgs",
                solver_scheme=scheme,
                substeps=8 if scheme == "tgs" else 2,
                solver_iterations=1 if scheme == "tgs" else 4,
                velocity_iterations=2,
                prepare_refresh_stride=1,
                sor_boost=1.0,
                mass_splitting_color_group_size=3,
                velocity_readout="substep_end",
                mass_splitting=True,
                contact_chunk_size=0 if scheme == "tgs" else 3,
                parallel_contact_prepare=True,
            )
            self.assertTrue(solver.world._enable_body_pair_grouping)
            samples = []
            for _ in range(5):
                pipeline.collide(state, contacts, dt=0.001)
                count = int(contacts.rigid_contact_count.numpy()[0])
                self.assertGreater(count, 1)
                if reverse:
                    wp.launch(
                        _reverse_alternate_contacts,
                        dim=64,
                        inputs=[
                            contacts.rigid_contact_count,
                            contacts.rigid_contact_shape0,
                            contacts.rigid_contact_shape1,
                            contacts.rigid_contact_point0,
                            contacts.rigid_contact_point1,
                            contacts.rigid_contact_margin0,
                            contacts.rigid_contact_margin1,
                            contacts.rigid_contact_normal,
                        ],
                        device=model.device,
                    )
                state.clear_forces()
                solver.step(state, state, model.control(), contacts, 0.001)
                solver.update_contacts(contacts)
                forces = contacts.force.numpy()[:count].copy()
                if scheme == "tgs":
                    # Compare moments about a common origin before reversing endpoints.
                    bodies = model.shape_body.numpy()[contacts.rigid_contact_shape0.numpy()[:count]]
                    centers = state.body_q.numpy()[bodies, :3]
                    forces[:, 3:] += np.cross(centers, forces[:, :3])
                if reverse:
                    forces[1::2] *= -1
                samples.append((state.body_q.numpy(), state.body_qd.numpy(), forces))
                np.testing.assert_allclose(_total_momentum(model, state), initial_momentum, atol=3e-5, rtol=0)
            results.append(samples)
        for reference, reversed_rows in zip(*results, strict=True):
            for expected, actual in zip(reference, reversed_rows, strict=True):
                np.testing.assert_allclose(actual, expected, atol=2e-6, rtol=1e-6)


if __name__ == "__main__":
    unittest.main()
