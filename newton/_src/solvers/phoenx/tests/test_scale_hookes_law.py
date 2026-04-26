# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Hooke's-law check for a scale built on a soft prismatic-joint limit.

A platform (or platform + cube grid) sits on a +z prismatic joint with
a PD spring-damper at the lower limit. At static equilibrium the joint
must compress past the limit by exactly ``F_total / limit_ke``, where
``F_total`` is the platform's own weight plus any cubes resting on it
(rotary Hooke's law applied to the linear DoF).

A miscalibrated limit-spring formula (e.g. wrong sign, missing
gravity, double-counting damping) shows up as a non-1:1
deflection-vs-load slope, which this test catches.
"""

from __future__ import annotations

import math
import unittest

import numpy as np
import warp as wp

import newton

from newton._src.solvers.phoenx.examples.example_phoenx_scale import (
    Example as ScaleExample,
    LIMIT_KE,
    LIMIT_KD,
    LIMIT_LOWER,
    PLATFORM_MASS,
    CUBE_MASS,
    CUBE_GRID,
    GRAVITY,
)


class _NullViewer:
    """Stub viewer so the example builds without a real GL window."""

    ui = None

    def set_model(self, m):
        pass

    def set_camera(self, **kw):
        pass

    def begin_frame(self, t):
        pass

    def log_state(self, s):
        pass

    def log_contacts(self, c, s):
        pass

    def end_frame(self):
        pass

    def register_ui_callback(self, *a, **kw):
        pass

    def apply_forces(self, s):
        pass


def _build_isolated_scale(
    *, platform_mass: float, n_cubes: int, cube_mass: float, limit_ke: float, limit_kd: float
):
    """Standalone scale builder for the parametric Hooke's-law sweep --
    avoids the example's hard-coded mass/cube count so the test can
    sweep across loads."""
    mb = newton.ModelBuilder()
    mb.add_ground_plane()
    cfg_static = newton.ModelBuilder.ShapeConfig(density=0.0)
    plat_hx, plat_hy, plat_hz = 0.5, 0.5, 0.025
    ixx = platform_mass / 3.0 * (plat_hy ** 2 + plat_hz ** 2)
    iyy = platform_mass / 3.0 * (plat_hx ** 2 + plat_hz ** 2)
    izz = platform_mass / 3.0 * (plat_hx ** 2 + plat_hy ** 2)
    plat = mb.add_link(
        xform=wp.transform(p=wp.vec3(0.0, 0.0, 0.20), q=wp.quat_identity()),
        mass=platform_mass,
        inertia=((ixx, 0, 0), (0, iyy, 0), (0, 0, izz)),
    )
    mb.add_shape_box(plat, hx=plat_hx, hy=plat_hy, hz=plat_hz, cfg=cfg_static)
    joint = mb.add_joint_prismatic(
        parent=-1,
        child=plat,
        parent_xform=wp.transform(p=wp.vec3(0.0, 0.0, 0.20), q=wp.quat_identity()),
        child_xform=wp.transform_identity(),
        axis=(0.0, 0.0, 1.0),
        limit_lower=LIMIT_LOWER,
        limit_upper=0.05,
        limit_ke=limit_ke,
        limit_kd=limit_kd,
    )
    mb.add_articulation([joint])

    if n_cubes > 0:
        cube_he = 0.07
        cixx = cube_mass / 3.0 * (cube_he ** 2 + cube_he ** 2)
        cinertia = ((cixx, 0, 0), (0, cixx, 0), (0, 0, cixx))
        grid = int(round(math.sqrt(n_cubes)))
        spacing = 2.2 * cube_he
        x0 = -((grid - 1) * spacing) * 0.5
        for i in range(grid):
            for j in range(grid):
                cube = mb.add_body(
                    xform=wp.transform(
                        p=wp.vec3(
                            x0 + i * spacing,
                            x0 + j * spacing,
                            0.20 + plat_hz + cube_he + 0.005,
                        ),
                        q=wp.quat_identity(),
                    ),
                    mass=cube_mass,
                    inertia=cinertia,
                )
                mb.add_shape_box(cube, hx=cube_he, hy=cube_he, hz=cube_he, cfg=cfg_static)

    mb.gravity = -GRAVITY
    model = mb.finalize()
    cp = newton.CollisionPipeline(model, contact_matching="sticky")
    contacts = cp.contacts()
    solver = newton.solvers.SolverPhoenX(model, substeps=16, solver_iterations=32, velocity_iterations=1)
    return model, solver, cp, contacts


def _settle_and_read_q(model, solver, cp, contacts, n_frames: int) -> tuple[float, float]:
    """Run ``n_frames`` steps; return final ``(joint_q[0], joint_qd[0])``."""
    s0 = model.state()
    s1 = model.state()
    newton.eval_fk(model, model.joint_q, model.joint_qd, s0)
    control = model.control()
    for _ in range(n_frames):
        s0.clear_forces()
        model.collide(s0, contacts=contacts, collision_pipeline=cp)
        solver.step(s0, s1, control, contacts, 1.0 / 60.0)
        s0, s1 = s1, s0
    jq = wp.zeros(model.joint_coord_count, dtype=wp.float32, device=model.device)
    jqd = wp.zeros(model.joint_dof_count, dtype=wp.float32, device=model.device)
    newton.eval_ik(model, s0, jq, jqd)
    return float(jq.numpy()[0]), float(jqd.numpy()[0])


@unittest.skipUnless(
    wp.get_preferred_device().is_cuda,
    "PhoenX scale test runs on CUDA only.",
)
class TestScaleHookesLaw(unittest.TestCase):
    """Linear Hooke's law on a scale: deflection past the lower limit
    must equal ``F_total / limit_ke`` at static equilibrium, across a
    sweep of loads."""

    def test_deflection_matches_hookes_law_across_loads(self) -> None:
        """Sweep platform / cube counts; verify deflection is linear in
        total weight with slope 1 / limit_ke. A miscalibrated spring
        (factor-of-2 error, missed gravity sign, etc.) shows up as a
        non-unit slope."""
        configs = [
            (5.0, 0, 0.0),
            (5.0, 1, 0.5),
            (5.0, 9, 0.5),
            (5.0, 25, 0.5),
            (50.0, 0, 0.0),
        ]
        for plat_m, n_cubes, cube_m in configs:
            with self.subTest(plat_m=plat_m, n_cubes=n_cubes, cube_m=cube_m):
                model, solver, cp, contacts = _build_isolated_scale(
                    platform_mass=plat_m,
                    n_cubes=n_cubes,
                    cube_mass=cube_m,
                    limit_ke=LIMIT_KE,
                    limit_kd=LIMIT_KD,
                )
                q, qd = _settle_and_read_q(model, solver, cp, contacts, n_frames=600)

                # Steady state: joint velocity below 1 mm/s.
                self.assertLess(
                    abs(qd),
                    0.01,
                    msg=f"joint not settled: qd={qd:.5f} m/s (try more substeps or kd)",
                )

                f_total = (plat_m + n_cubes * cube_m) * GRAVITY
                expected_deflection = f_total / LIMIT_KE
                actual_deflection = LIMIT_LOWER - q
                # Tolerance: 5% relative or 0.1 mm absolute, whichever is larger.
                tol = max(0.05 * expected_deflection, 1.0e-4)
                self.assertAlmostEqual(
                    actual_deflection,
                    expected_deflection,
                    delta=tol,
                    msg=f"plat_m={plat_m}, n_cubes={n_cubes}: "
                        f"deflection={actual_deflection*1000:.3f} mm vs "
                        f"expected {expected_deflection*1000:.3f} mm "
                        f"(F_total={f_total:.2f} N, ke={LIMIT_KE:.0f} N/m)",
                )

    def test_stiffness_sweep_inverse_proportional(self) -> None:
        """Doubling ``limit_ke`` halves the deflection (the linear
        Hooke's law signature). Catches non-linear or capped spring
        responses."""
        configs = [50_000.0, 100_000.0, 200_000.0, 500_000.0]
        plat_m = 50.0  # keeps the deflection well above float32 noise
        deflections = []
        for ke in configs:
            kd = 2.0 * math.sqrt(ke * plat_m)  # critical damping
            model, solver, cp, contacts = _build_isolated_scale(
                platform_mass=plat_m,
                n_cubes=0,
                cube_mass=0.0,
                limit_ke=ke,
                limit_kd=kd,
            )
            q, qd = _settle_and_read_q(model, solver, cp, contacts, n_frames=600)
            self.assertLess(abs(qd), 0.01, msg=f"ke={ke}: not settled")
            deflections.append((ke, LIMIT_LOWER - q))
        # ke * deflection should be ~ constant (= F_total).
        f_expected = plat_m * GRAVITY
        for ke, dx in deflections:
            product = ke * dx
            self.assertAlmostEqual(
                product,
                f_expected,
                delta=0.05 * f_expected,
                msg=f"ke={ke}: ke*dx = {product:.2f} N vs expected {f_expected:.2f} N",
            )

    def test_full_scale_example_settles_at_hookes_law(self) -> None:
        """The actual ``example_phoenx_scale.Example`` (5x5 cube grid on
        a 5 kg platform) must settle at the predicted deflection."""
        ex = ScaleExample(_NullViewer(), None)
        # 1200 frames @ 60 Hz = 20 s; the cubes need a few seconds to
        # settle on the platform before the platform itself reaches a
        # clean steady state.
        for _ in range(1200):
            ex.step()

        # Read the prismatic joint's q via eval_ik on the example's model.
        jq = wp.zeros(ex.model.joint_coord_count, dtype=wp.float32, device=ex.device)
        jqd = wp.zeros(ex.model.joint_dof_count, dtype=wp.float32, device=ex.device)
        newton.eval_ik(ex.model, ex.state_0, jq, jqd)
        q = float(jq.numpy()[0])
        qd = float(jqd.numpy()[0])
        self.assertLess(abs(qd), 0.01, msg=f"example not settled: qd={qd:.5f} m/s")

        f_total = (PLATFORM_MASS + CUBE_GRID * CUBE_GRID * CUBE_MASS) * GRAVITY
        expected_deflection = f_total / LIMIT_KE
        actual_deflection = LIMIT_LOWER - q
        # 10% tolerance for the contact-coupled test (cubes sliding slightly etc.).
        self.assertAlmostEqual(
            actual_deflection,
            expected_deflection,
            delta=0.10 * expected_deflection,
            msg=f"example deflection={actual_deflection*1000:.3f} mm vs expected "
                f"{expected_deflection*1000:.3f} mm "
                f"(F_total={f_total:.2f} N)",
        )


if __name__ == "__main__":
    wp.init()
    unittest.main()
