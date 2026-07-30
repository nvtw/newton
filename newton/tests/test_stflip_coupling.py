# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import unittest

import numpy as np
import warp as wp

import newton


class TestSTFLIPCoupling(unittest.TestCase):
    def _devices(self):
        """Return every device used for coupling validation."""
        devices = ["cpu"]
        if wp.is_cuda_available():
            devices.append("cuda:0")
        return devices

    def _create_contact_case(self, device):
        """Create one off-center particle contact with a dynamic box."""
        builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
        newton.solvers.SolverSTFLIP.register_custom_attributes(builder)
        builder.default_shape_cfg.ke = 4000.0
        builder.default_shape_cfg.kd = 20.0
        builder.default_shape_cfg.kf = 0.0
        builder.default_shape_cfg.mu = 0.0
        body = builder.add_body(
            xform=wp.transform(wp.vec3(0.0), wp.quat_identity()),
            mass=2.0,
            inertia=wp.mat33(np.eye(3, dtype=np.float32) * 0.05),
            label="coupled box",
        )
        builder.add_shape_box(body, hx=0.2, hy=0.2, hz=0.2)
        builder.add_particle(
            pos=(0.22, 0.1, 0.0),
            vel=(0.0, 0.0, 0.0),
            mass=0.05,
            radius=0.05,
        )
        model = builder.finalize(device=device)
        model.soft_contact_ke = 4000.0
        model.soft_contact_kd = 20.0
        model.soft_contact_kf = 0.0
        model.soft_contact_mu = 0.0
        collision_pipeline = newton.CollisionPipeline(
            model,
            broad_phase="nxn",
            soft_contact_margin=0.1,
        )
        contacts = collision_pipeline.contacts()
        solver = newton.solvers.SolverSTFLIP(
            model,
            newton.solvers.SolverSTFLIP.Config(
                cell_size=0.1,
                tile_size=4,
                max_active_tile_count=27,
                padding_tiles=1,
                pressure_iterations=16,
                transfer_scheme="pic",
                temporal_staggering=False,
            ),
        )
        return model, solver, collision_pipeline, contacts, body

    def _body_force_for_particle_sampling(self, device, particle_count):
        """Measure one contact patch with fixed total particle mass."""
        builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
        newton.solvers.SolverSTFLIP.register_custom_attributes(builder)
        body = builder.add_body(
            xform=wp.transform(wp.vec3(0.0), wp.quat_identity()),
            mass=2.0,
            inertia=wp.mat33(np.eye(3, dtype=np.float32) * 0.05),
        )
        builder.add_shape_box(body, hx=0.2, hy=0.2, hz=0.2)
        for _ in range(particle_count):
            builder.add_particle(
                pos=(0.22, 0.0, 0.0),
                vel=(-1.0, 0.0, 0.0),
                mass=0.4 / particle_count,
                radius=0.05,
            )
        model = builder.finalize(device=device)
        collision_pipeline = newton.CollisionPipeline(model, broad_phase="nxn", soft_contact_margin=0.1)
        contacts = collision_pipeline.contacts()
        solver = newton.solvers.SolverSTFLIP(
            model,
            newton.solvers.SolverSTFLIP.Config(
                cell_size=0.1,
                tile_size=4,
                max_active_tile_count=27,
                padding_tiles=1,
                pressure_iterations=4,
                transfer_scheme="pic",
                temporal_staggering=False,
            ),
        )
        state_in = model.state()
        state_out = model.state()
        state_in.clear_forces()
        collision_pipeline.collide(state_in, contacts)
        solver.step(state_in, state_out, None, contacts, 1.0 / 240.0)
        return state_in.body_f.numpy()[body, 0]

    def test_contact_force_and_wrench_are_equal_and_opposite(self):
        """Balance particle force and rigid wrench at an off-center contact."""
        for device in self._devices():
            with self.subTest(device=device):
                model, solver, collision_pipeline, contacts, body = self._create_contact_case(device)
                state_in = model.state()
                state_out = model.state()
                state_in.clear_forces()
                collision_pipeline.collide(state_in, contacts)
                self.assertGreater(int(contacts.soft_contact_count.numpy()[0]), 0)
                solver.step(state_in, state_out, None, contacts, 1.0 / 240.0)

                particle_force = state_in.particle_f.numpy()[0]
                body_wrench = state_in.body_f.numpy()[body]
                self.assertGreater(float(np.linalg.norm(particle_force)), 1.0)
                np.testing.assert_allclose(
                    particle_force + body_wrench[:3],
                    np.zeros(3),
                    rtol=2.0e-6,
                    atol=2.0e-5,
                )

                particle_position = state_in.particle_q.numpy()[0]
                body_position = state_in.body_q.numpy()[body, :3]
                total_torque = (
                    np.cross(particle_position, particle_force)
                    + np.cross(body_position, body_wrench[:3])
                    + body_wrench[3:]
                )
                np.testing.assert_allclose(total_torque, np.zeros(3), rtol=2.0e-6, atol=2.0e-5)

    def test_contact_moves_particle_and_dynamic_body_apart(self):
        """Move both sides of a penetrating fluid-rigid contact apart."""
        for device in self._devices():
            with self.subTest(device=device):
                model, solver, collision_pipeline, contacts, body = self._create_contact_case(device)
                state_in = model.state()
                state_out = model.state()
                state_in.clear_forces()
                collision_pipeline.collide(state_in, contacts)
                solver.step(state_in, state_out, None, contacts, 1.0 / 240.0)

                particle_velocity = state_out.particle_qd.numpy()[0]
                body_velocity = state_out.body_qd.numpy()[body, :3]
                self.assertGreater(float(particle_velocity[0]), 0.0)
                self.assertLess(float(body_velocity[0]), 0.0)
                self.assertTrue(np.all(np.isfinite(state_out.particle_q.numpy())))
                self.assertTrue(np.all(np.isfinite(state_out.body_q.numpy())))

    def test_contact_impulse_is_particle_sampling_invariant(self):
        """Keep rigid coupling nearly invariant under particle refinement."""
        for device in self._devices():
            with self.subTest(device=device):
                coarse = self._body_force_for_particle_sampling(device, 4)
                fine = self._body_force_for_particle_sampling(device, 64)
                self.assertLess(coarse, 0.0)
                self.assertLess(fine, 0.0)
                self.assertAlmostEqual(coarse / fine, 1.0, delta=0.06)


if __name__ == "__main__":
    unittest.main()
