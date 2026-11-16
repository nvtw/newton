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


if __name__ == "__main__":
    unittest.main()
