# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Check exact articulation contact response on CPU."""

import unittest

import numpy as np
import warp as wp

import newton
from newton.solvers import SolverBase, SolverPhoenX


class TestDirectContactsCPU(unittest.TestCase):
    def test_slider_free_fall_and_ground_support(self):
        """Preserve free slider motion without contacts and support its weight with a floor."""
        with wp.ScopedDevice("cpu"):
            for floor in (False, True):
                with self.subTest(floor=floor):
                    builder = newton.ModelBuilder(gravity=(0, 0, -9.81))
                    cfg = builder.ShapeConfig(mu=0, mu_static=0, density=1000)
                    body = builder.add_link(xform=wp.transform(wp.vec3(0, 0, 0.1)))
                    builder.add_shape_box(body, hx=0.1, hy=0.1, hz=0.1, cfg=cfg)
                    joint = builder.add_joint_prismatic(
                        -1,
                        body,
                        axis=newton.Axis.Z,
                        parent_xform=wp.transform(wp.vec3(0, 0, 0.1)),
                        target_ke=0,
                        target_kd=0,
                        limit_lower=-1e6,
                        limit_upper=1e6,
                    )
                    builder.add_articulation([joint])
                    if floor:
                        builder.add_ground_plane(cfg=cfg)
                    model = builder.finalize()
                    solver = SolverPhoenX(
                        model,
                        joint_mode="maximal_direct",
                        step_layout="single_world",
                        substeps=1,
                        solver_iterations=4,
                        contact_hertz=80,
                    )
                    state, control = model.state(), model.control()
                    newton.eval_fk(model, model.joint_q, model.joint_qd, state)
                    contacts = model._collision_pipeline.contacts()
                    observables = solver.observables({SolverBase.ObservableKind.CONTACT_F}, contacts=contacts)
                    forces = []
                    dt, steps = 1 / 120, 60
                    for frame in range(steps):
                        solver.collide(state, contacts, dt=dt)
                        solver.step(state, state, control, contacts, dt, observables=observables)
                        if frame >= 30:
                            count = int(contacts.rigid_contact_count.numpy()[0])
                            forces.append(abs(float(observables.contact_f.numpy()[:count, 2].sum())))
                    z = float(state.body_q.numpy()[body, 2])
                    vz = float(state.body_qd.numpy()[body, 2])
                    if floor:
                        self.assertAlmostEqual(z, 0.1, delta=0.001)
                        self.assertLess(abs(vz), 0.001)
                        self.assertAlmostEqual(np.mean(forces), float(model.body_mass.numpy()[body]) * 9.81, delta=0.01)
                    else:
                        self.assertAlmostEqual(vz, -9.81 * steps * dt, delta=1e-5)
                        self.assertAlmostEqual(z, 0.1 - 0.5 * 9.81 * dt**2 * steps * (steps + 1), delta=1e-5)


@unittest.skipUnless(wp.is_cuda_available(), "Graph-captured direct contacts require CUDA")
class TestDirectContactsCappedDrive(unittest.TestCase):
    def test_capped_drive_presses_light_probe_into_wall(self):
        """A 50 N stiff drive presses a 10 g probe into a wall without launching it.

        The owned contacts first stop the probe against the uncapped spring. When the
        cap then engages, the contacts must answer the capped motor in the same pass;
        otherwise their leftover impulse threw the probe back at thousands of m/s.
        """
        with wp.ScopedDevice("cuda:0"):
            builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
            builder.default_shape_cfg = builder.ShapeConfig(gap=0.002, density=0.0)
            builder.add_shape_box(-1, xform=wp.transform(wp.vec3(0.061, 0.0, 0.0)), hx=0.05, hy=0.05, hz=0.05)
            mass = 0.01
            probe = builder.add_link(mass=mass, inertia=wp.mat33(np.eye(3) * mass * 0.02**2 / 6.0))
            builder.add_shape_box(probe, hx=0.01, hy=0.01, hz=0.01)
            joint = builder.add_joint_prismatic(
                -1, probe, axis=(1.0, 0.0, 0.0), target_ke=1.0e6, target_kd=1.0e3, effort_limit=50.0, target_pos=0.0
            )
            builder.add_articulation([joint])
            model = builder.finalize()
            control = model.control()
            # The target lies 11 mm inside the wall, whose face is 1 mm ahead of the probe.
            control.joint_target_q.fill_(0.012)
            solver = SolverPhoenX(model, joint_mode="maximal_direct", substeps=1)
            state_0, state_1 = model.state(), model.state()
            newton.eval_fk(model, model.joint_q, model.joint_qd, state_0)
            contacts = solver._collision_pipeline.contacts()
            dt = 1.0 / 240.0

            def step():
                nonlocal state_0, state_1
                solver.collide(state_0, contacts, dt=dt)
                solver.step(state_0, state_1, control, contacts, dt)
                state_0, state_1 = state_1, state_0

            step()
            step()  # warm every kernel; two swaps return the original buffers
            with wp.ScopedCapture() as capture:
                step()
                step()
            for _ in range(59):
                wp.capture_launch(capture.graph)
            x = float(state_0.body_q.numpy()[probe, 0])
            v = float(state_0.body_qd.numpy()[probe, 0])
        self.assertTrue(np.isfinite(x) and np.isfinite(v))
        # The face sits on the wall up to the soft-contact compliance at 500 g, and at rest.
        self.assertLess(abs(x + 0.01 - 0.011), 0.011)
        self.assertLess(abs(v), 0.05)


if __name__ == "__main__":
    unittest.main()
