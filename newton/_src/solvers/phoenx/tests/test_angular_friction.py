# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Torsional and rolling contact friction follow Newton's moment-length coefficients."""

import unittest

import numpy as np
import warp as wp

import newton

_RADIUS = 0.1
_MASS = 1.0
_DT = 1.0 / 480.0


def _sphere_on_plane(mu_torsional: float, mu_rolling: float):
    cfg = newton.ModelBuilder.ShapeConfig(mu=0.0, mu_torsional=mu_torsional, mu_rolling=mu_rolling)
    builder = newton.ModelBuilder()
    builder.add_ground_plane(cfg=cfg)
    body = builder.add_body(xform=wp.transform((0.0, 0.0, _RADIUS), wp.quat_identity()))
    cfg.density = _MASS / (4.0 / 3.0 * np.pi * _RADIUS**3)
    builder.add_shape_sphere(body, radius=_RADIUS, cfg=cfg)
    model = builder.finalize()
    pipeline = newton.CollisionPipeline(model, contact_matching="sticky")
    solver = newton.solvers.SolverPhoenX(model, collision_pipeline=pipeline, substeps=1, solver_iterations=8)
    return model, pipeline, solver


def _run(model, pipeline, solver, torque, spin, steps):
    state, control, contacts = model.state(), model.control(), pipeline.contacts()
    qd = state.body_qd.numpy()
    qd[0, 3:] = spin
    state.body_qd.assign(qd)
    force = np.zeros((1, 6), dtype=np.float32)
    force[0, 3:] = torque

    def step():
        state.body_f.assign(force)
        pipeline.collide(state, contacts)
        solver.step(state, state, control, contacts, _DT)

    step()
    with wp.ScopedCapture() as capture:
        step()
    for _ in range(steps):
        wp.capture_launch(capture.graph)
    return state.body_qd.numpy()[0, 3:]


@unittest.skipUnless(wp.is_cuda_available(), "requires CUDA graph capture")
class TestAngularFriction(unittest.TestCase):
    def test_torsional_moment_holds_and_yields(self):
        """Hold a twist below ``mu_torsional * N`` and spin up at the excess above it."""
        mu_t = 0.02
        limit = mu_t * _MASS * 9.81
        inertia = 0.4 * _MASS * _RADIUS**2
        with wp.ScopedDevice("cuda:0"):
            model, pipeline, solver = _sphere_on_plane(mu_t, 0.0)
            held = _run(model, pipeline, solver, (0.0, 0.0, 0.5 * limit), (0.0, 0.0, 0.0), 240)
            model, pipeline, solver = _sphere_on_plane(mu_t, 0.0)
            moving = _run(model, pipeline, solver, (0.0, 0.0, 2.0 * limit), (0.0, 0.0, 0.0), 240)
        self.assertLess(abs(held[2]), 1.0e-3)
        # 242 steps of net torque `limit` on the sphere's inertia.
        np.testing.assert_allclose(moving[2], limit / inertia * 242 * _DT, rtol=0.05)

    def test_rolling_ball_decelerates_at_the_rolling_moment(self):
        """Slow a rolling ball by ``mu_rolling * N`` about its rolling axis."""
        mu_r = 0.002
        with wp.ScopedDevice("cuda:0"):
            model, pipeline, solver = _sphere_on_plane(0.0, mu_r)
            # Pure spin about x with friction-free sliding: the sphere spins down in place.
            spin = _run(model, pipeline, solver, (0.0, 0.0, 0.0), (5.0, 0.0, 0.0), 240)
            model, pipeline, solver = _sphere_on_plane(0.0, 0.0)
            free = _run(model, pipeline, solver, (0.0, 0.0, 0.0), (5.0, 0.0, 0.0), 240)
        self.assertAlmostEqual(free[0], 5.0, delta=1.0e-3)
        # Frictionless sliding: only the spin's own inertia resists the rolling moment.
        decel = mu_r * _MASS * 9.81 / (0.4 * _MASS * _RADIUS**2)
        np.testing.assert_allclose(spin[0], 5.0 - decel * 242 * _DT, rtol=0.05)


if __name__ == "__main__":
    unittest.main()
