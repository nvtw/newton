# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Contact friction follows each shape's :class:`~newton.ShapeFrictionMode` and moment-length coefficients."""

import unittest

import numpy as np
import warp as wp

import newton
from newton import ShapeFrictionMode

_RADIUS = 0.1
_MASS = 1.0
_DT = 1.0 / 480.0
_STEPS = 240
_INERTIA = 0.4 * _MASS * _RADIUS**2


def _sphere_on_plane(
    sphere_mode: ShapeFrictionMode,
    ground_mode: ShapeFrictionMode | None = None,
    mu: float = 0.0,
    mu_torsional: float = 0.0,
    mu_rolling: float = 0.0,
):
    """Return a solver for a sphere resting on a ground plane; the ground's mode defaults to the sphere's."""
    cfg = newton.ModelBuilder.ShapeConfig(
        mu=mu,
        mu_torsional=mu_torsional,
        mu_rolling=mu_rolling,
        friction_mode=sphere_mode if ground_mode is None else ground_mode,
    )
    builder = newton.ModelBuilder()
    builder.add_ground_plane(cfg=cfg)
    body = builder.add_body(xform=wp.transform((0.0, 0.0, _RADIUS), wp.quat_identity()))
    cfg.density = _MASS / (4.0 / 3.0 * np.pi * _RADIUS**3)
    cfg.friction_mode = sphere_mode
    builder.add_shape_sphere(body, radius=_RADIUS, cfg=cfg)
    model = builder.finalize()
    pipeline = newton.CollisionPipeline(model, contact_matching="sticky")
    solver = newton.solvers.SolverPhoenX(model, collision_pipeline=pipeline, substeps=1, solver_iterations=8)
    return model, pipeline, solver


def _run(model, pipeline, solver, velocity=(0.0, 0.0, 0.0), spin=(0.0, 0.0, 0.0), torque=(0.0, 0.0, 0.0)):
    """Step ``_STEPS + 2`` times from the given velocity and spin; return the final ``(v, w)``."""
    state, control, contacts = model.state(), model.control(), pipeline.contacts()
    qd = state.body_qd.numpy()
    qd[0, :3] = velocity
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
    for _ in range(_STEPS):
        wp.capture_launch(capture.graph)
    qd = state.body_qd.numpy()[0]
    return qd[:3], qd[3:]


@unittest.skipUnless(wp.is_cuda_available(), "requires CUDA graph capture")
class TestAngularFriction(unittest.TestCase):
    def test_none_mode_slides_without_friction(self):
        """Slide freely when both shapes are ``NONE``; one ``SLIDING`` shape enables friction."""
        with wp.ScopedDevice("cuda:0"):
            free, _ = _run(*_sphere_on_plane(ShapeFrictionMode.NONE, mu=1.0), velocity=(1.0, 0.0, 0.0))
            union, _ = _run(
                *_sphere_on_plane(ShapeFrictionMode.NONE, ShapeFrictionMode.SLIDING, mu=1.0), velocity=(1.0, 0.0, 0.0)
            )
        self.assertAlmostEqual(free[0], 1.0, delta=1.0e-3)
        # Friction turns the slide into rolling at 5/7 of the initial speed.
        self.assertAlmostEqual(union[0], 5.0 / 7.0, delta=0.02)

    def test_sliding_mode_keeps_spinning(self):
        """Ignore ``mu_torsional`` and ``mu_rolling`` when no shape enables them."""
        with wp.ScopedDevice("cuda:0"):
            model, pipeline, solver = _sphere_on_plane(
                ShapeFrictionMode.SLIDING, mu=0.5, mu_torsional=0.02, mu_rolling=0.002
            )
            _, spin = _run(model, pipeline, solver, spin=(0.0, 0.0, 5.0))
        self.assertEqual(solver.world._contact_cols.angular_impulse.shape[0], 0)
        self.assertAlmostEqual(spin[2], 5.0, delta=1.0e-3)

    def test_torsional_mode_damps_spin_by_union(self):
        """Spin down at ``mu_torsional * N`` when only the sphere enables torsional friction."""
        mu_t = 0.02
        with wp.ScopedDevice("cuda:0"):
            model, pipeline, solver = _sphere_on_plane(
                ShapeFrictionMode.SLIDING_TORSIONAL, ShapeFrictionMode.SLIDING, mu=0.5, mu_torsional=mu_t
            )
            _, spin = _run(model, pipeline, solver, spin=(0.0, 0.0, 30.0))
        decel = mu_t * _MASS * 9.81 / _INERTIA
        np.testing.assert_allclose(spin[2], 30.0 - decel * (_STEPS + 2) * _DT, rtol=0.05)

    def test_rolling_mode_damps_rolling_ball(self):
        """Slow a ball rolling without slip only when a shape enables rolling friction."""
        speed = 1.0
        mu_r = 0.002
        results = {}
        with wp.ScopedDevice("cuda:0"):
            for mode in (ShapeFrictionMode.SLIDING, ShapeFrictionMode.SLIDING_ROLLING):
                model, pipeline, solver = _sphere_on_plane(mode, mu=1.0, mu_rolling=mu_r)
                results[mode], _ = _run(model, pipeline, solver, (speed, 0.0, 0.0), (0.0, speed / _RADIUS, 0.0))
        self.assertAlmostEqual(results[ShapeFrictionMode.SLIDING][0], speed, delta=2.0e-3)
        # Rolling without slip: the moment mu_r * N acts on the inertia m * r^2 + I about the contact point.
        decel = mu_r * _MASS * 9.81 * _RADIUS / (_MASS * _RADIUS**2 + _INERTIA)
        np.testing.assert_allclose(
            results[ShapeFrictionMode.SLIDING_ROLLING][0], speed - decel * (_STEPS + 2) * _DT, rtol=0.05
        )

    def test_torsional_moment_holds_and_yields(self):
        """Hold a twist below ``mu_torsional * N`` and spin up at the excess above it."""
        mu_t = 0.02
        limit = mu_t * _MASS * 9.81
        mode = ShapeFrictionMode.SLIDING_TORSIONAL
        with wp.ScopedDevice("cuda:0"):
            _, held = _run(*_sphere_on_plane(mode, mu_torsional=mu_t), torque=(0.0, 0.0, 0.5 * limit))
            _, moving = _run(*_sphere_on_plane(mode, mu_torsional=mu_t), torque=(0.0, 0.0, 2.0 * limit))
        self.assertLess(abs(held[2]), 1.0e-3)
        # 242 steps of net torque `limit` on the sphere's inertia.
        np.testing.assert_allclose(moving[2], limit / _INERTIA * (_STEPS + 2) * _DT, rtol=0.05)

    def test_rolling_ball_decelerates_at_the_rolling_moment(self):
        """Slow a frictionless sphere's spin about a tangent axis by ``mu_rolling * N``."""
        mu_r = 0.002
        with wp.ScopedDevice("cuda:0"):
            _, spin = _run(*_sphere_on_plane(ShapeFrictionMode.SLIDING_ROLLING, mu_rolling=mu_r), spin=(5.0, 0.0, 0.0))
        # Frictionless sliding: only the spin's own inertia resists the rolling moment.
        decel = mu_r * _MASS * 9.81 / _INERTIA
        np.testing.assert_allclose(spin[0], 5.0 - decel * (_STEPS + 2) * _DT, rtol=0.05)

    def test_cloth_slides_freely_on_none_shape(self):
        """Cloth counts as ``NONE``: it slides freely on a ``NONE`` box and slows on a ``SLIDING`` one."""
        speeds = {}
        with wp.ScopedDevice("cuda:0"):
            for mode in (ShapeFrictionMode.NONE, ShapeFrictionMode.SLIDING):
                builder = newton.ModelBuilder(gravity=(0.0, 0.0, -9.81))
                cfg = newton.ModelBuilder.ShapeConfig(mu=1.0, friction_mode=mode)
                builder.add_shape_box(body=-1, hx=0.5, hy=0.5, hz=0.05, cfg=cfg)
                builder.add_cloth_grid(
                    pos=wp.vec3(-0.45, -0.1, 0.06),
                    rot=wp.quat_identity(),
                    vel=wp.vec3(0.5, 0.0, 0.0),
                    dim_x=1,
                    dim_y=1,
                    cell_x=0.2,
                    cell_y=0.2,
                    mass=0.1,
                    particle_radius=0.01,
                )
                model = builder.finalize()
                pipeline = newton.CollisionPipeline(
                    model, contact_matching="sticky", enable_rigid_soft_full_surface_contact=True
                )
                solver = newton.solvers.SolverPhoenX(
                    model, collision_pipeline=pipeline, step_layout="single_world", substeps=2, solver_iterations=4
                )
                state, contacts = model.state(), pipeline.contacts()

                def step(solver=solver, state=state, contacts=contacts):
                    solver.collide(state, contacts)
                    solver.step(state, state, None, contacts, 1.0 / 60.0)

                step()
                with wp.ScopedCapture() as capture:
                    step()
                for _ in range(30):
                    wp.capture_launch(capture.graph)
                speeds[mode] = float(state.particle_qd.numpy()[:, 0].mean())
        self.assertAlmostEqual(speeds[ShapeFrictionMode.NONE], 0.5, delta=0.02)
        self.assertLess(speeds[ShapeFrictionMode.SLIDING], 0.25)


if __name__ == "__main__":
    unittest.main()
