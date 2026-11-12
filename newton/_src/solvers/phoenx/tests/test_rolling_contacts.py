# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Rolling and spinning bodies keep their support and their motion.

A step finds its contacts once, at its starting pose. A wheel or ball turns within the step,
so its contact must not ride up the body with the material point it started at.
"""

import unittest

import numpy as np
import warp as wp

import newton

_RADIUS = 0.08  # [m]


def _ground_builder(mu: float) -> newton.ModelBuilder:
    builder = newton.ModelBuilder()
    builder.add_ground_plane(cfg=newton.ModelBuilder.ShapeConfig(mu=mu))
    return builder


def _run(model, solver, pipeline, state, control, frames: int, record):
    contacts = pipeline.contacts()
    with wp.ScopedCapture() as capture:
        pipeline.collide(state, contacts)
        solver.step(state, state, control, contacts, 1.0 / 60.0)
    samples = []
    for _ in range(frames):
        wp.capture_launch(capture.graph)
        samples.append(record(state))
    return np.asarray(samples)


def _wheel_cylinder_xform() -> wp.transform:
    """Cylinder axis along y, so the wheel rolls along x."""
    return wp.transform(wp.vec3(), wp.quat_from_axis_angle(wp.vec3(1.0, 0.0, 0.0), 0.5 * np.pi))


@unittest.skipUnless(wp.is_cuda_available(), "PhoenX tests run on CUDA with graph capture")
class TestRollingContacts(unittest.TestCase):
    def test_spinning_cylinder_keeps_support_and_spin(self):
        """A frictionless cylinder spinning about its axis on the ground stays at rest height and spin."""
        builder = _ground_builder(mu=0.0)
        body = builder.add_body(xform=wp.transform(wp.vec3(0.0, 0.0, _RADIUS), wp.quat_identity()))
        cfg = newton.ModelBuilder.ShapeConfig(density=1000.0, mu=0.0)
        builder.add_shape_cylinder(body, xform=_wheel_cylinder_xform(), radius=_RADIUS, half_height=0.025, cfg=cfg)
        model = builder.finalize(device="cuda:0")
        state, control = model.state(), model.control()
        spin = 7.5  # [rad/s]
        velocity = state.body_qd.numpy()
        velocity[body] = [0.0, 0.0, 0.0, 0.0, spin, 0.0]
        state.body_qd.assign(velocity)
        pipeline = newton.CollisionPipeline(model, contact_matching="sticky")
        solver = newton.solvers.SolverPhoenX(model, collision_pipeline=pipeline, substeps=4, solver_iterations=4)
        samples = _run(
            model,
            solver,
            pipeline,
            state,
            control,
            120,
            lambda s: (s.body_q.numpy()[body, 2], s.body_qd.numpy()[body, 4]),
        )
        height, final_spin = samples[60:, 0], samples[-1, 1]
        # Turning a cylinder about its own axis changes neither its contact nor its momentum.
        self.assertGreater(float(height.min()), _RADIUS - 0.5e-3, f"sank to {height.min():.5f} m")
        self.assertAlmostEqual(float(final_spin), spin, delta=0.02 * spin)

    def test_rolling_ball_keeps_rolling(self):
        """A ball rolling without slip on a flat floor keeps its speed and its height."""
        radius = 0.1
        builder = _ground_builder(mu=1.0)
        body = builder.add_body(xform=wp.transform(wp.vec3(0.0, 0.0, radius), wp.quat_identity()))
        builder.add_shape_sphere(body, radius=radius, cfg=newton.ModelBuilder.ShapeConfig(density=1000.0, mu=1.0))
        model = builder.finalize(device="cuda:0")
        state, control = model.state(), model.control()
        speed = 3.0  # [m/s]
        velocity = state.body_qd.numpy()
        velocity[body] = [speed, 0.0, 0.0, 0.0, speed / radius, 0.0]
        state.body_qd.assign(velocity)
        pipeline = newton.CollisionPipeline(model, contact_matching="sticky")
        solver = newton.solvers.SolverPhoenX(model, collision_pipeline=pipeline, substeps=4, solver_iterations=4)
        samples = _run(
            model,
            solver,
            pipeline,
            state,
            control,
            120,
            lambda s: (s.body_q.numpy()[body, 2], s.body_qd.numpy()[body, 0]),
        )
        self.assertGreater(float(samples[:, 0].min()), radius - 0.5e-3, f"dipped to {samples[:, 0].min():.5f} m")
        self.assertAlmostEqual(float(samples[-1, 1]), speed, delta=0.01 * speed)


if __name__ == "__main__":
    unittest.main()
