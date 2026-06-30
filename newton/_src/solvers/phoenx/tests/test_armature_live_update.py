# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Live joint_armature update tests for SolverPhoenX.

Maximal PhoenX stores armature in an auxiliary inertial row. Domain
randomization therefore rebuilds the joint-row data when joint DOF properties
change, without mutating body inertia. Coverage includes direct row readback and
graph-captured period measurements before and after a host-side model update.
"""

from __future__ import annotations

import unittest

import numpy as np
import warp as wp

import newton
from newton._src.solvers.flags import SolverNotifyFlags
from newton._src.solvers.phoenx.constraints.constraint_joint import _OFF_ARMATURE


def _build_anchored_revolute(*, mass: float, length: float, armature: float) -> newton.Model:
    """Single revolute joint with one body, world-anchored. Bob at
    ``-z * length``, axis ``+y`` so the body swings in the ``x-z``
    plane under ``-z`` gravity."""
    mb = newton.ModelBuilder(up_axis=newton.Axis.Z)
    newton.solvers.SolverMuJoCo.register_custom_attributes(mb)
    mb.default_joint_cfg = newton.ModelBuilder.JointDofConfig(armature=armature)
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


@unittest.skipUnless(
    wp.get_preferred_device().is_cuda and wp.is_mempool_enabled(wp.get_preferred_device()),
    "PhoenX armature tests run on CUDA with graph capture only",
)
class TestArmatureLiveUpdate(unittest.TestCase):
    """Live updates rewrite the joint row without mutating body inertia."""

    @staticmethod
    def _row_armature(solver: newton.solvers.SolverPhoenX) -> float:
        return float(solver.world.constraints.data.numpy()[int(_OFF_ARMATURE), 0])

    def test_joint_dof_properties_notify_updates_row(self) -> None:
        model = _build_anchored_revolute(mass=1.0, length=1.0, armature=0.0)
        solver = newton.solvers.SolverPhoenX(model, substeps=1)
        inertia_before = solver.bodies.inverse_inertia.numpy().copy()

        model.joint_armature.assign(np.array([2.0], dtype=np.float32))
        solver.notify_model_changed(int(SolverNotifyFlags.JOINT_DOF_PROPERTIES))

        self.assertAlmostEqual(self._row_armature(solver), 2.0)
        np.testing.assert_array_equal(solver.bodies.inverse_inertia.numpy(), inertia_before)

    def test_repeated_notify_is_idempotent(self) -> None:
        model = _build_anchored_revolute(mass=1.0, length=1.0, armature=0.0)
        solver = newton.solvers.SolverPhoenX(model, substeps=1)
        model.joint_armature.assign(np.array([1.5], dtype=np.float32))
        solver.notify_model_changed(int(SolverNotifyFlags.JOINT_DOF_PROPERTIES))
        first = solver.world.constraints.data.numpy().copy()
        solver.notify_model_changed(int(SolverNotifyFlags.JOINT_DOF_PROPERTIES))
        second = solver.world.constraints.data.numpy()
        np.testing.assert_array_equal(second, first)

    def test_revert_to_zero_updates_row(self) -> None:
        model = _build_anchored_revolute(mass=1.0, length=1.0, armature=5.0)
        solver = newton.solvers.SolverPhoenX(model, substeps=1)
        self.assertAlmostEqual(self._row_armature(solver), 5.0)
        model.joint_armature.assign(np.array([0.0], dtype=np.float32))
        solver.notify_model_changed(int(SolverNotifyFlags.JOINT_DOF_PROPERTIES))
        self.assertEqual(self._row_armature(solver), 0.0)


def _run_pendulum_graph(
    solver: newton.solvers.SolverPhoenX,
    model: newton.Model,
    *,
    frames: int,
    dt: float,
    init_angle: float = 0.05,
) -> np.ndarray:
    """Capture (clear, step) into a CUDA graph and replay ``frames`` times,
    returning the joint-angle history. Pinned to a single solver instance
    so subsequent notifications + recaptures probe the live-update path."""
    device = wp.get_device()
    assert device.is_cuda, "graph-capture pendulum requires CUDA"

    s0 = model.state()
    s1 = model.state()
    control = model.control()
    s0.joint_q.assign(np.array([float(init_angle)], dtype=np.float32))
    s0.joint_qd.assign(np.array([0.0], dtype=np.float32))
    newton.eval_fk(model, s0.joint_q, s0.joint_qd, s0)

    history = np.empty(frames, dtype=np.float32)
    jq = wp.zeros(model.joint_coord_count, dtype=wp.float32, device=model.device)
    jqd = wp.zeros(model.joint_dof_count, dtype=wp.float32, device=model.device)

    def _frame() -> None:
        s0.clear_forces()
        solver.step(s0, s1, control, None, dt)
        # eval_ik copies the post-step joint state out without aliasing
        # ``s1.joint_q`` (which PhoenX overwrote via its own eval_ik in
        # step). Mirror the rigid state back into ``s0`` so the next
        # frame starts from the integrated configuration.
        newton.eval_ik(model, s1, jq, jqd)
        wp.copy(s0.body_q, s1.body_q)
        wp.copy(s0.body_qd, s1.body_qd)

    if frames < 1:
        return history

    # Warm-up: compile every kernel + allocate scratch.
    _frame()
    history[0] = float(jq.numpy()[0])
    if frames == 1:
        return history

    with wp.ScopedCapture(device=device) as capture:
        _frame()
    graph = capture.graph

    for i in range(1, frames):
        wp.capture_launch(graph)
        history[i] = float(jq.numpy()[0])
    return history


@unittest.skipUnless(wp.get_preferred_device().is_cuda, "PhoenX armature tests run on CUDA only")
class TestArmaturePeriodAfterLiveUpdate(unittest.TestCase):
    """End-to-end check: a graph-captured pendulum's period must
    track the live-updated armature. The period formula is
    ``T = 2*pi * sqrt((I_chain + a) / (m*g*L))`` with
    ``I_chain = m*L^2 + I_com``."""

    def test_period_changes_after_live_armature_update(self) -> None:
        mass = 1.0
        length = 1.0
        g = 9.81
        dt = 1.0 / 400.0
        I_chain = mass * length * length + 1.0e-4
        frames = 4800  # ~12 s at dt=2.5ms; >= 3 periods even with armature=2.

        # Baseline solver instance with armature=0.
        model = _build_anchored_revolute(mass=mass, length=length, armature=0.0)
        solver = newton.solvers.SolverPhoenX(
            model,
            substeps=8,
            solver_iterations=8,
            velocity_iterations=1,
        )

        h0 = _run_pendulum_graph(solver, model, frames=frames, dt=dt)
        T0 = _measure_period_zero_crossings(h0, dt)
        T_expected_0 = 2.0 * np.pi * np.sqrt(I_chain / (mass * g * length))
        rel_err_0 = abs(T0 - T_expected_0) / T_expected_0
        self.assertLess(
            rel_err_0,
            0.05,
            f"baseline armature=0: T_sim={T0:.4f} s vs T_expected={T_expected_0:.4f} s",
        )

        # Live update: bump armature to a non-zero value. notify_model_changed
        # is host-side; capture the next run after the notify completes.
        new_armature = 2.0
        model.joint_armature.assign(np.array([new_armature], dtype=np.float32))
        solver.notify_model_changed(int(SolverNotifyFlags.JOINT_DOF_PROPERTIES))

        h1 = _run_pendulum_graph(solver, model, frames=frames, dt=dt)
        T1 = _measure_period_zero_crossings(h1, dt)
        T_expected_1 = 2.0 * np.pi * np.sqrt((I_chain + new_armature) / (mass * g * length))
        rel_err_1 = abs(T1 - T_expected_1) / T_expected_1
        self.assertLess(
            rel_err_1,
            0.05,
            (
                f"post-update armature={new_armature}: T_sim={T1:.4f} s vs "
                f"T_expected={T_expected_1:.4f} s (a regression that skips the "
                "live row refresh would still report ~T_expected_0)"
            ),
        )

        # Period must have actually moved -- the failure mode the fix
        # addresses is silent: the simulation runs, but the armature
        # value never propagates. T1 should be strictly greater than T0
        # by a factor of sqrt((I+a)/I), which here is ~sqrt(3) ~ 1.73.
        ratio = T1 / T0
        expected_ratio = np.sqrt((I_chain + new_armature) / I_chain)
        rel_err_ratio = abs(ratio - expected_ratio) / expected_ratio
        self.assertLess(
            rel_err_ratio,
            0.05,
            (
                f"period ratio T1/T0 = {ratio:.4f} vs expected sqrt((I+a)/I) = "
                f"{expected_ratio:.4f} (rel err {rel_err_ratio:.2%}). A regression "
                "that skips the live row refresh would yield ratio ~1.0."
            ),
        )


if __name__ == "__main__":
    unittest.main()
