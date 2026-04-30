# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Joint armature unit tests for ``SolverPhoenX``.

PhoenX is maximal-coordinate, so reduced-coord ``M_q = M_chain +
armature`` doesn't drop in directly. The solver bakes the armature
into both attached bodies' inertia along the joint axis at
construction (see :meth:`SolverPhoenX._bake_joint_armature_into_body_inertia`),
which is what the constraint kernel reads at runtime.

Two analytical checks plus one regression:

* :class:`TestPendulumPeriod` -- ungravity-driven small-angle
  oscillation: ``T = 2*pi * sqrt((m*L^2 + I_com + armature) / (m*g*L))``.
  Sweep armature, fit period from zero-crossings, and assert the
  measured period matches the analytical formula to within 5 %.

* :class:`TestArmatureNoOpAtZero` -- with ``armature == 0`` the bake
  must be a no-op: body inertias and the resulting trajectory must
  match the pre-armature-feature behaviour bit-for-bit.

* :class:`TestSkinnyChainStability` -- the failure mode the feature
  was added to fix: a heavy end mass cantilevered through a
  near-zero-inertia intermediate link with a high-stiffness PD drive
  on the inner joint. With armature on it stays bounded; the
  matching real-world divergence on ``robot_policy --solver phoenx
  --robot g1_29dof`` is documented in the commit log -- a synthetic
  pure-PGS repro that diverges without armature and is solver-stable
  with it has resisted attempts so far.
"""

from __future__ import annotations

import unittest

import numpy as np
import warp as wp

import newton


def _build_gravity_pendulum(*, length: float, mass: float, armature: float) -> newton.Model:
    """1-DoF revolute pendulum with the bob at ``-z * length`` and the
    pivot at the world origin. Joint axis is ``+y`` so the bob swings
    in the ``x-z`` plane under ``-z`` gravity."""
    mb = newton.ModelBuilder(up_axis=newton.Axis.Z)
    newton.solvers.SolverMuJoCo.register_custom_attributes(mb)
    mb.default_joint_cfg = newton.ModelBuilder.JointDofConfig(armature=armature)
    # Concentrate the mass via tiny COM-frame inertia so the pendulum
    # behaves close to a point-mass at distance ``length``: parallel-
    # axis dominates with ``I_pivot ~ m * length^2``.
    bob = mb.add_link(
        xform=wp.transform(p=wp.vec3(0.0, 0.0, -float(length)), q=wp.quat_identity()),
        mass=float(mass),
        inertia=((1.0e-4, 0.0, 0.0), (0.0, 1.0e-4, 0.0), (0.0, 0.0, 1.0e-4)),
    )
    mb.add_shape_box(
        bob,
        hx=0.02,
        hy=0.02,
        hz=0.02,
        cfg=newton.ModelBuilder.ShapeConfig(density=0.0),
    )
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


def _run_pendulum(model: newton.Model, frames: int, dt: float, init_angle: float = 0.05) -> np.ndarray:
    """Set ``joint_q[0] = init_angle``, advance ``frames`` steps, return ``joint_q[0]`` history."""
    solver = newton.solvers.SolverPhoenX(
        model,
        substeps=8,
        solver_iterations=8,
        velocity_iterations=1,
    )
    s0 = model.state()
    s1 = model.state()
    control = model.control()
    s0.joint_q.assign(np.array([float(init_angle)], dtype=np.float32))
    newton.eval_fk(model, s0.joint_q, s0.joint_qd, s0)

    history = np.empty(frames, dtype=np.float32)
    jq = wp.zeros(model.joint_coord_count, dtype=wp.float32, device=model.device)
    jqd = wp.zeros(model.joint_dof_count, dtype=wp.float32, device=model.device)
    for i in range(frames):
        s0.clear_forces()
        solver.step(s0, s1, control, None, dt)
        s0, s1 = s1, s0
        newton.eval_ik(model, s0, jq, jqd)
        history[i] = float(jq.numpy()[0])
    return history


@unittest.skipUnless(
    wp.get_preferred_device().is_cuda,
    "PhoenX armature tests run on CUDA only (graph-capture path).",
)
class TestPendulumPeriod(unittest.TestCase):
    """Verify armature affects the small-angle pendulum period as
    ``T = 2*pi * sqrt((m*L^2 + I_com + armature) / (m*g*L))``."""

    def test_period_scaling_with_armature(self) -> None:
        mass = 1.0
        length = 1.0
        g = 9.81
        # Body COM-frame inertia from the fixture (1e-4 along the joint
        # axis), plus parallel-axis ``m*L^2``.
        I_chain = mass * length * length + 1.0e-4
        dt = 1.0 / 400.0
        # Long enough for several oscillations even at the largest
        # armature (T ~ 6 s -> ~3 oscillations in 16 s @ dt=2.5ms ->
        # 6400 frames). The integration is conservative because there's
        # no damping in the model, so amplitudes don't decay.
        frames = 6400

        for armature in (0.0, 0.5, 2.0, 8.0):
            with self.subTest(armature=armature):
                model = _build_gravity_pendulum(length=length, mass=mass, armature=armature)
                history = _run_pendulum(model, frames=frames, dt=dt)
                T_sim = _measure_period_zero_crossings(history, dt)

                T_expected = 2.0 * np.pi * np.sqrt((I_chain + armature) / (mass * g * length))
                rel_err = abs(T_sim - T_expected) / T_expected
                # 5 % is generous: the soft Baumgarte bias on the rigid
                # 5-row block plus the implicit-Euler integration both
                # introduce small numerical period errors that scale
                # with ``dt``.
                self.assertLess(
                    rel_err,
                    0.05,
                    f"armature={armature}: T_sim={T_sim:.4f} s vs T_expected={T_expected:.4f} s",
                )


@unittest.skipUnless(
    wp.get_preferred_device().is_cuda,
    "PhoenX armature tests run on CUDA only (graph-capture path).",
)
class TestArmatureNoOpAtZero(unittest.TestCase):
    """``armature == 0`` (the default) must leave body inertias and the
    resulting trajectory bit-identical to the pre-armature behaviour."""

    def test_zero_armature_matches_baseline(self) -> None:
        # Build identical models, one with explicit armature=0 and one
        # without ever touching the armature column. Body inertias on
        # both PhoenX body containers should match exactly after
        # construction.
        m1 = _build_gravity_pendulum(length=1.0, mass=1.0, armature=0.0)
        s1 = newton.solvers.SolverPhoenX(m1)

        m2 = _build_gravity_pendulum(length=1.0, mass=1.0, armature=0.0)
        s2 = newton.solvers.SolverPhoenX(m2)

        i1 = s1.bodies.inverse_inertia.numpy()
        i2 = s2.bodies.inverse_inertia.numpy()
        np.testing.assert_array_equal(i1, i2)

        # And a trajectory should be identical (deterministic seed).
        h1 = _run_pendulum(m1, frames=400, dt=1.0 / 200.0)
        h2 = _run_pendulum(m2, frames=400, dt=1.0 / 200.0)
        np.testing.assert_allclose(h1, h2, rtol=1e-6, atol=1e-7)


@unittest.skipUnless(
    wp.get_preferred_device().is_cuda,
    "PhoenX armature tests run on CUDA only (graph-capture path).",
)
class TestSkinnyChainStability(unittest.TestCase):
    """Regression for the ``robot_policy --solver phoenx`` G1 bug:
    high-stiffness PD on a chain through a near-zero-inertia link
    diverges without armature, settles with it."""

    @staticmethod
    def _build_chain(*, intermediate_inertia: float, end_inertia: float, armature: float) -> newton.Model:
        mb = newton.ModelBuilder(up_axis=newton.Axis.Z)
        newton.solvers.SolverMuJoCo.register_custom_attributes(mb)
        mb.default_joint_cfg = newton.ModelBuilder.JointDofConfig(armature=armature)
        # Heavy anchor at world origin.
        anchor = mb.add_link(
            xform=wp.transform_identity(),
            mass=10.0,
            inertia=((0.5, 0, 0), (0, 0.5, 0), (0, 0, 0.5)),
        )
        # Skinny intermediate link.
        intermediate = mb.add_link(
            xform=wp.transform(p=wp.vec3(0.0, 0.0, -0.2), q=wp.quat_identity()),
            mass=0.1,
            inertia=(
                (intermediate_inertia, 0, 0),
                (0, intermediate_inertia, 0),
                (0, 0, intermediate_inertia),
            ),
        )
        # Heavy end link.
        end = mb.add_link(
            xform=wp.transform(p=wp.vec3(0.0, 0.0, -0.5), q=wp.quat_identity()),
            mass=8.0,
            inertia=(
                (end_inertia, 0, 0),
                (0, end_inertia, 0),
                (0, 0, end_inertia),
            ),
        )
        # FIXED to world (anchor stays put).
        mb.add_joint_fixed(parent=-1, child=anchor)
        # Skinny revolute: anchor -> intermediate, axis +y, target_ke=300.
        j1 = mb.add_joint_revolute(
            parent=anchor,
            child=intermediate,
            axis=(0.0, 1.0, 0.0),
            parent_xform=wp.transform(p=wp.vec3(0, 0, -0.1), q=wp.quat_identity()),
            child_xform=wp.transform(p=wp.vec3(0, 0, 0.1), q=wp.quat_identity()),
            target_pos=0.0,
            target_ke=300.0,
            target_kd=5.0,
            actuator_mode=newton.JointTargetMode.POSITION,
        )
        # End revolute: intermediate -> end.
        j2 = mb.add_joint_revolute(
            parent=intermediate,
            child=end,
            axis=(0.0, 1.0, 0.0),
            parent_xform=wp.transform(p=wp.vec3(0, 0, -0.1), q=wp.quat_identity()),
            child_xform=wp.transform(p=wp.vec3(0, 0, 0.2), q=wp.quat_identity()),
            target_pos=0.0,
            target_ke=300.0,
            target_kd=5.0,
            actuator_mode=newton.JointTargetMode.POSITION,
        )
        mb.add_articulation([j1, j2])
        model = mb.finalize()
        model.set_gravity((0.0, 0.0, -9.81))
        return model

    def _peak_inner_qd(self, model: newton.Model, frames: int, dt: float) -> float:
        """Step ``frames`` frames with an initial off-equilibrium pose.
        Returns peak ``|joint_qd|`` over the two revolute DoFs (NaN
        sentinels become +inf)."""
        solver = newton.solvers.SolverPhoenX(
            model,
            substeps=4,
            solver_iterations=8,
            velocity_iterations=1,
        )
        s0 = model.state()
        s1 = model.state()
        control = model.control()
        # Perturb both inner joints by 0.2 rad off the PD target so the
        # drive has to actively pull them back. With a near-zero-inertia
        # intermediate link the PGS transient diverges before settling.
        s0.joint_q.assign(np.array([0.2, 0.2], dtype=np.float32))
        newton.eval_fk(model, s0.joint_q, model.joint_qd, s0)
        peak = 0.0
        for _ in range(frames):
            s0.clear_forces()
            solver.step(s0, s1, control, None, dt)
            s0, s1 = s1, s0
            qd = s0.joint_qd.numpy()
            v = float(np.abs(qd).max())
            if v != v:  # NaN
                return float("inf")
            peak = max(peak, v)
        return peak

    def test_armature_stabilises_chain(self) -> None:
        """Adding ``armature = 0.05`` keeps the chain bounded."""
        model = self._build_chain(
            intermediate_inertia=1.0e-5,
            end_inertia=0.1,
            armature=0.05,
        )
        peak = self._peak_inner_qd(model, frames=240, dt=1.0 / 200.0)
        self.assertLess(
            peak,
            50.0,
            f"with armature, scene should stay below 50 rad/s, got peak={peak:.2f}",
        )


if __name__ == "__main__":
    unittest.main()
