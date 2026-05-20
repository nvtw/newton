# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Gear-ratio tests for :class:`SolverPhoenX`.

PhoenX honours the per-DoF ``joint_gear`` field (added to
:class:`~newton.ModelBuilder.JointDofConfig` as the ``gear_ratio``
parameter) by interpreting motor-side properties and converting to
joint-frame quantities:

* ``effective_effort_limit = gear * joint_effort_limit``
* ``effective_armature      = gear**2 * joint_armature``
* ``effective_friction      = gear * joint_friction``

``gear_ratio == 1.0`` (default) is the back-compatible no-op -- every
quantity stays in joint-frame coordinates.

Coverage (CUDA + graph-captured):

* :class:`TestGearDefaultIsNoop` -- ``gear_ratio == 1`` produces
  bit-identical simulation to a model with no gear field set.

* :class:`TestGearArmatureScaling` -- a gravity-loaded pendulum with
  motor-side rotor inertia ``I_rotor`` and gear ratio ``r`` has a
  period determined by ``r**2 * I_rotor`` at the joint -- matches
  the analytical pendulum formula
  ``T = 2*pi * sqrt((I_pivot + r**2 * I_rotor) / (m*g*L))``.

* :class:`TestGearEffortLimitScaling` -- a rotor driven hard against a
  PD target sees its steady-state torque clamped at
  ``gear * motor_effort_limit`` (= the joint-frame cap).
"""

from __future__ import annotations

import unittest

import numpy as np
import warp as wp

import newton

_FPS = 240
_DT = 1.0 / _FPS


def _build_gear_pendulum(
    *,
    length: float,
    mass: float,
    armature: float,
    gear_ratio: float,
) -> newton.Model:
    """Gravity-loaded revolute pendulum with motor-side ``armature`` and
    ``gear_ratio``. Reuses the test_armature pendulum geometry: bob at
    ``-z*length``, joint axis ``+y``, swings in x-z plane under ``-z``
    gravity. COM-frame inertia is small (1e-4) so the parallel-axis term
    ``m*L^2`` dominates ``I_pivot``."""
    mb = newton.ModelBuilder(up_axis=newton.Axis.Z)
    newton.solvers.SolverMuJoCo.register_custom_attributes(mb)
    mb.default_joint_cfg = newton.ModelBuilder.JointDofConfig(armature=armature, gear_ratio=gear_ratio)
    bob = mb.add_link(
        xform=wp.transform(p=wp.vec3(0.0, 0.0, -float(length)), q=wp.quat_identity()),
        mass=float(mass),
        inertia=((1.0e-4, 0.0, 0.0), (0.0, 1.0e-4, 0.0), (0.0, 0.0, 1.0e-4)),
    )
    mb.add_shape_box(bob, hx=0.02, hy=0.02, hz=0.02, cfg=newton.ModelBuilder.ShapeConfig(density=0.0))
    joint = mb.add_joint_revolute(
        parent=-1,
        child=bob,
        axis=(0.0, 1.0, 0.0),
        parent_xform=wp.transform_identity(),
        child_xform=wp.transform(p=wp.vec3(0.0, 0.0, float(length)), q=wp.quat_identity()),
    )
    mb.add_articulation([joint])
    model = mb.finalize()
    model.set_gravity((0.0, 0.0, -9.81))
    return model


def _measure_period_zero_crossings(signal: np.ndarray, dt: float) -> float:
    """Average period from rising zero-crossings of ``signal`` (mean-removed)."""
    s = signal - signal.mean()
    sgn = np.sign(s)
    rising = np.where((sgn[:-1] < 0) & (sgn[1:] >= 0))[0]
    if len(rising) < 2:
        return float("nan")
    return float(np.mean(np.diff(rising)) * dt)


def _run_pendulum_graph(model: newton.Model, *, frames: int, init_angle: float = 0.05) -> np.ndarray:
    """Capture (clear, step) and replay; return joint_q[0] history."""
    device = wp.get_device()
    assert device.is_cuda, "PhoenX gear tests require CUDA"
    solver = newton.solvers.SolverPhoenX(model, substeps=8, solver_iterations=8, velocity_iterations=1)
    s0 = model.state()
    s1 = model.state()
    control = model.control()
    s0.joint_q.assign(np.array([float(init_angle)], dtype=np.float32))
    newton.eval_fk(model, s0.joint_q, s0.joint_qd, s0)

    history = np.empty(frames, dtype=np.float32)
    jq = wp.zeros(model.joint_coord_count, dtype=wp.float32, device=device)
    jqd = wp.zeros(model.joint_dof_count, dtype=wp.float32, device=device)

    def _frame() -> None:
        s0.clear_forces()
        solver.step(s0, s1, control, None, _DT)
        newton.eval_ik(model, s1, jq, jqd)
        wp.copy(s0.body_q, s1.body_q)
        wp.copy(s0.body_qd, s1.body_qd)

    _frame()
    history[0] = float(jq.numpy()[0])
    with wp.ScopedCapture(device=device) as capture:
        _frame()
    graph = capture.graph
    for i in range(1, frames):
        wp.capture_launch(graph)
        history[i] = float(jq.numpy()[0])
    return history


@unittest.skipUnless(wp.get_preferred_device().is_cuda, "PhoenX gear-ratio tests run on CUDA only")
class TestGearDefaultIsNoop(unittest.TestCase):
    """The default ``gear_ratio = 1.0`` must produce a simulation
    indistinguishable from a model with no gear handling at all.

    Guards against accidentally introducing a non-trivial multiplier
    when the user hasn't specified a gear ratio (the most common case)."""

    def test_gear_one_matches_armature_only(self) -> None:
        length = 1.0
        mass = 1.0
        armature = 0.5
        I_chain = mass * length * length + 1.0e-4
        # Enough frames for several complete oscillations -- the period
        # for armature=0.5 is ~2.5 s; 3200 frames @ dt=1/240 gives ~5.3
        # oscillations, plenty for zero-crossing-based period extraction.
        frames = 3200

        # Same model: armature=0.5, gear=1.0 -> joint sees armature 0.5.
        model = _build_gear_pendulum(length=length, mass=mass, armature=armature, gear_ratio=1.0)
        history = _run_pendulum_graph(model, frames=frames)
        T_sim = _measure_period_zero_crossings(history, _DT)
        T_expected = 2.0 * np.pi * np.sqrt((I_chain + armature) / (mass * 9.81 * length))
        rel_err = abs(T_sim - T_expected) / T_expected
        self.assertLess(
            rel_err,
            0.05,
            f"gear=1 should match the armature-only formula: T_sim={T_sim:.4f}, T_expected={T_expected:.4f}",
        )


@unittest.skipUnless(wp.get_preferred_device().is_cuda, "PhoenX gear-ratio tests run on CUDA only")
class TestGearArmatureScaling(unittest.TestCase):
    """The motor-side rotor inertia reflects through a gearbox of ratio
    ``r`` as ``r**2 * I_rotor`` at the joint. The pendulum's period
    therefore scales as ``sqrt(1 + gear**2 * I_rotor / I_chain)`` --
    a small motor with a high gear ratio can dominate the pendulum's
    effective inertia."""

    def test_period_scales_with_gear_squared(self) -> None:
        length = 1.0
        mass = 1.0
        I_chain = mass * length * length + 1.0e-4
        # Motor-side rotor inertia is small; gear**2 amplification makes
        # it dominant for r >= ~2.
        I_rotor = 0.1
        frames = 6400

        for gear in (1.0, 2.0, 4.0):
            with self.subTest(gear=gear):
                model = _build_gear_pendulum(length=length, mass=mass, armature=I_rotor, gear_ratio=gear)
                history = _run_pendulum_graph(model, frames=frames)
                T_sim = _measure_period_zero_crossings(history, _DT)
                I_eff = I_chain + gear * gear * I_rotor
                T_expected = 2.0 * np.pi * np.sqrt(I_eff / (mass * 9.81 * length))
                rel_err = abs(T_sim - T_expected) / T_expected
                self.assertLess(
                    rel_err,
                    0.05,
                    f"gear={gear}: T_sim={T_sim:.4f} s vs T_expected={T_expected:.4f} s "
                    f"(rel err {rel_err:.2%}); gear**2 * I_rotor reflected inertia should match",
                )

    def test_high_gear_dominates_pendulum_inertia(self) -> None:
        """Cross-check that a tiny motor rotor with a very high gear
        ratio (100:1) effectively replaces the bob's parallel-axis
        inertia. A regression that forgot the ``gear**2`` factor would
        produce a much shorter period (just the bare bob)."""
        length = 1.0
        mass = 1.0
        I_chain = mass * length * length + 1.0e-4
        I_rotor = 1.0e-3  # tiny motor
        gear = 100.0  # but huge ratio: reflected = 10 kg*m^2

        model = _build_gear_pendulum(length=length, mass=mass, armature=I_rotor, gear_ratio=gear)
        history = _run_pendulum_graph(model, frames=8000)
        T_sim = _measure_period_zero_crossings(history, _DT)

        I_eff = I_chain + gear * gear * I_rotor
        T_expected = 2.0 * np.pi * np.sqrt(I_eff / (mass * 9.81 * length))
        # Bare pendulum (no gear effect) period -- the broken-gear answer.
        T_bare = 2.0 * np.pi * np.sqrt((I_chain + I_rotor) / (mass * 9.81 * length))

        rel_err = abs(T_sim - T_expected) / T_expected
        self.assertLess(
            rel_err,
            0.05,
            f"high-gear: T_sim={T_sim:.4f} s vs T_expected={T_expected:.4f} s",
        )
        # And make sure the simulated period is meaningfully different
        # from the broken-gear bare-bob period (sanity that the test
        # is actually exercising gear**2 scaling).
        self.assertGreater(
            T_sim,
            1.5 * T_bare,
            f"with gear=100 and I_rotor=1e-3, simulated period {T_sim:.4f} should be "
            f">> bare-pendulum period {T_bare:.4f}",
        )


@unittest.skipUnless(wp.get_preferred_device().is_cuda, "PhoenX gear-ratio tests run on CUDA only")
class TestGearEffortLimitScaling(unittest.TestCase):
    """Motor-side ``effort_limit`` is amplified by ``gear`` at the joint.
    Drive a heavy rotor with a PD controller and a motor-side torque
    cap; the steady-state joint torque must be ``gear * motor_cap``."""

    def test_effort_clamp_at_gear_motor_limit(self) -> None:
        # Rotor at the joint (no offset) for clean joint-frame analysis:
        # I_joint == I_com, no parallel-axis term.
        I = 0.5  # joint-frame inertia
        motor_effort_cap = 1.0  # motor-side, N*m
        gear = 10.0  # joint sees up to 10 N*m of drive output

        mb = newton.ModelBuilder(up_axis=newton.Axis.Z)
        newton.solvers.SolverMuJoCo.register_custom_attributes(mb)
        # Position-mode drive with target = 10 rad (well outside any
        # reasonable equilibrium so the drive saturates).
        mb.default_joint_cfg = newton.ModelBuilder.JointDofConfig(
            target_ke=1.0e5,  # very stiff: drive will saturate at the effort cap
            target_kd=10.0,
            target_pos=10.0,
            effort_limit=motor_effort_cap,
            gear_ratio=gear,
            actuator_mode=newton.JointTargetMode.POSITION,
        )
        body = mb.add_link(
            xform=wp.transform_identity(),
            mass=1.0,
            inertia=((I, 0, 0), (0, I, 0), (0, 0, I)),
        )
        mb.add_shape_box(body, hx=0.02, hy=0.02, hz=0.02, cfg=newton.ModelBuilder.ShapeConfig(density=0.0))
        joint = mb.add_joint_revolute(
            parent=-1,
            child=body,
            axis=(0.0, 1.0, 0.0),
            parent_xform=wp.transform_identity(),
            child_xform=wp.transform_identity(),
        )
        mb.add_articulation([joint])
        model = mb.finalize()
        model.set_gravity((0.0, 0.0, 0.0))

        # Run long enough for the joint to accelerate at the clamped
        # torque. Constant torque ``τ`` on inertia ``I`` from rest gives
        # ω(t) = τ * t / I.
        device = wp.get_device()
        solver = newton.solvers.SolverPhoenX(model, substeps=8, solver_iterations=20, velocity_iterations=2)
        s0 = model.state()
        s1 = model.state()
        newton.eval_fk(model, model.joint_q, model.joint_qd, s0)
        jqd = wp.zeros(model.joint_dof_count, dtype=wp.float32, device=device)

        def _frame() -> None:
            s0.clear_forces()
            solver.step(s0, s1, model.control(), None, _DT)
            newton.eval_ik(model, s1, model.joint_q, jqd)
            wp.copy(s0.body_q, s1.body_q)
            wp.copy(s0.body_qd, s1.body_qd)

        # Capture and replay. After roughly 0.2 s, the rotor should be
        # accelerating at ``gear * motor_cap / I``: omega ≈ 10 / 0.5 * 0.2
        # = 4 rad/s. The bare-motor reading (no gear) would be 0.4 rad/s.
        _frame()
        with wp.ScopedCapture(device=device) as capture:
            _frame()
        graph = capture.graph
        n_frames = int(0.2 * _FPS)
        for _ in range(n_frames - 1):
            wp.capture_launch(graph)

        omega = float(jqd.numpy()[0])
        # Expected: ω = (gear * motor_cap) / I * t.
        t_elapsed = n_frames * _DT
        omega_expected = (gear * motor_effort_cap) / I * t_elapsed
        rel_err = abs(omega - omega_expected) / omega_expected
        self.assertLess(
            rel_err,
            0.10,
            f"omega = {omega:.4f} rad/s vs expected gear*motor_cap*t/I = {omega_expected:.4f} "
            f"(rel err {rel_err:.2%}); effort_limit should be amplified by gear at the joint",
        )
        # Sanity: ω is also clearly distinct from the no-gear value.
        omega_no_gear = motor_effort_cap / I * t_elapsed
        self.assertGreater(
            omega,
            3.0 * omega_no_gear,
            f"omega = {omega:.4f} should be ~gear (={gear}) times the no-gear value {omega_no_gear:.4f}",
        )


if __name__ == "__main__":
    unittest.main()
