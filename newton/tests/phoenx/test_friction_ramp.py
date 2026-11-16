# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Check Coulomb static thresholds through the public model material bridge."""

import math
import unittest

import numpy as np
import warp as wp

import newton
from newton.solvers import SolverPhoenX


class TestFrictionRamp(unittest.TestCase):
    def test_static_friction_from_model(self):
        """Hold a cube above the kinetic threshold using authored static friction."""
        with wp.ScopedDevice("cpu"):
            theta, half, dt = math.radians(20.0), 0.25, 1.0 / 240.0
            normal = np.array([math.sin(theta), 0.0, math.cos(theta)])
            downhill = np.array([math.cos(theta), 0.0, -math.sin(theta)])
            rotation = wp.quat_from_axis_angle(wp.vec3(0.0, 1.0, 0.0), theta)
            builder = newton.ModelBuilder()
            body = builder.add_body(
                xform=wp.transform(wp.vec3(normal * half), rotation),
                mass=1.0,
                inertia=wp.mat33(np.eye(3) / 24.0),
            )
            # tan(20 degrees) is between the kinetic and static coefficients.
            cfg = builder.ShapeConfig(density=0, mu=0.2, mu_static=0.6)
            builder.add_shape_box(body, hx=half, hy=half, hz=half, cfg=cfg)
            builder.add_shape_plane(xform=wp.transform(wp.vec3(0.0), rotation), cfg=cfg)
            model = builder.finalize(device="cpu")
            pipeline = newton.CollisionPipeline(model, rigid_contact_max=32, contact_matching="sticky")
            contacts = pipeline.contacts()
            solver = SolverPhoenX(
                model,
                collision_pipeline=pipeline,
                step_layout="single_world",
                substeps=1,
                solver_iterations=4,
                velocity_iterations=1,
                sleeping_velocity_threshold=0.0,
            )
            state, control = model.state(), model.control()
            offsets = []
            for _ in range(480):
                state.clear_forces()
                pipeline.collide(state, contacts, dt=dt)
                solver.step(state, state, control, contacts, dt)
                offsets.append(float(state.body_q.numpy()[body, :3] @ downhill))
            # Allow finite contact settling, but not kinetic sliding (~3 m).
            self.assertLess(abs(offsets[-1]), 0.001)
            self.assertLess(abs(offsets[-1] - offsets[240]), 0.0001)


if __name__ == "__main__":
    unittest.main()
