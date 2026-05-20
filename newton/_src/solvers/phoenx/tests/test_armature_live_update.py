# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Live ``joint_armature`` update tests for :class:`SolverPhoenX`.

PhoenX bakes joint armature into both attached bodies' inertia at
construction (see
:meth:`SolverPhoenX._bake_joint_armature_into_body_inertia`). Domain
randomization for sim-to-real transfer requires this bake to be
re-applied whenever the user mutates ``model.joint_armature`` between
episodes; the regression this guards is the bake stamping the very
first armature value into ``bodies.inverse_inertia`` and never
refreshing it.

The fix routes ``notify_model_changed(JOINT_DOF_PROPERTIES)`` through
``_launch_init_phoenx_bodies`` (which resets the inertia back to
``Model.body_inv_inertia``) followed by
``_bake_joint_armature_into_body_inertia`` (which adds the new
armature). Both passes are non-graph-capture-safe by design --
``notify_model_changed`` is documented as a host-side reconfigure.

Two layers of coverage, all CUDA + graph-capture only (PhoenX is
GPU-only; graph capture is the shipping execution mode for the
``step`` loop -- ``notify_model_changed`` is called *between*
captures):

* :class:`TestArmatureLiveUpdate` -- direct readback of
  ``solver.bodies.inverse_inertia`` before and after the notify.
  Pins the wiring without any physics: a regression that forgets to
  refresh would leave the post-notify inertia bit-identical to the
  pre-notify inertia.

* :class:`TestArmaturePeriodAfterLiveUpdate` -- end-to-end pendulum
  with graph-captured step loop. Measures period T0 with armature=0,
  bumps armature to 2.0 via ``model.joint_armature.assign(...)`` +
  notify, recaptures the graph, measures T1, and verifies
  ``T1/T0 == sqrt((I + a)/I)`` to within 5 %. Both periods are
  measured under graph capture (n_frames ~ 1000).
"""

from __future__ import annotations

import unittest

import numpy as np
import warp as wp

import newton
from newton._src.solvers.flags import SolverNotifyFlags


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


@unittest.skipUnless(wp.get_preferred_device().is_cuda, "PhoenX armature tests run on CUDA only")
class TestArmatureLiveUpdate(unittest.TestCase):
    """Direct readback of ``bodies.inverse_inertia`` before and after a
    ``notify_model_changed(JOINT_DOF_PROPERTIES)`` armature update."""

    def test_bake_refreshes_on_joint_dof_properties_notify(self) -> None:
        """Bumping ``model.joint_armature`` and notifying
        ``JOINT_DOF_PROPERTIES`` must propagate into the inertia bake.

        For a fixed-anchor revolute about ``+y`` the bake adds
        ``alpha * (axis_in_child) (axis_in_child)^T`` to the child's
        inertia along the joint axis (``alpha == armature`` when the
        parent is the world). The body's axial inverse-inertia
        therefore shifts as ``1 / (I_body + a)`` -- a regression that
        skips the refresh would leave ``1 / (I_body + a_initial)``.
        """
        mass = 1.0
        length = 1.0
        I_body_axial = 1.0e-4  # diagonal Y-component from the fixture

        model = _build_anchored_revolute(mass=mass, length=length, armature=0.0)
        solver = newton.solvers.SolverPhoenX(model, substeps=1)

        # Baseline: armature 0. PhoenX slot 1 is the bob (slot 0 is the
        # static world anchor).
        inv_I_before = solver.bodies.inverse_inertia.numpy()[1]
        # The joint axis (+y) -> diagonal[1] is the axial component.
        axial_inv_before = float(inv_I_before[1, 1])
        # Without armature the inverse along the joint axis is 1 / I_body.
        # Tolerance is loose because the bake's ``invI -> I -> invI``
        # round-trip with single precision drifts at the 1e-5 level.
        self.assertAlmostEqual(
            axial_inv_before,
            1.0 / I_body_axial,
            delta=1.0,
            msg=f"baseline axial inv-inertia should be 1/I_body, got {axial_inv_before}",
        )

        # Live update: bump armature to 2.0 and notify.
        new_armature = 2.0
        model.joint_armature.assign(np.array([new_armature], dtype=np.float32))
        solver.notify_model_changed(int(SolverNotifyFlags.JOINT_DOF_PROPERTIES))

        inv_I_after = solver.bodies.inverse_inertia.numpy()[1]
        axial_inv_after = float(inv_I_after[1, 1])
        # After bake: axial inverse-inertia should be 1 / (I_body + a).
        expected_after = 1.0 / (I_body_axial + new_armature)
        self.assertAlmostEqual(
            axial_inv_after,
            expected_after,
            delta=1.0e-2 * expected_after,
            msg=(
                f"after live armature update to {new_armature}, axial inv-inertia "
                f"should be {expected_after:.6f}, got {axial_inv_after:.6f}"
            ),
        )

        # Off-axis components must be unchanged (bake only touches the
        # axial outer product).
        np.testing.assert_allclose(
            inv_I_after[0, 0],
            inv_I_before[0, 0],
            rtol=1.0e-4,
            err_msg="off-axis (X) inv-inertia should be unchanged after axial-armature bake",
        )
        np.testing.assert_allclose(
            inv_I_after[2, 2],
            inv_I_before[2, 2],
            rtol=1.0e-4,
            err_msg="off-axis (Z) inv-inertia should be unchanged after axial-armature bake",
        )

    def test_repeated_notify_does_not_drift(self) -> None:
        """Calling ``notify_model_changed`` twice with the same armature
        must be idempotent: the bake resets inertia from
        ``Model.body_inv_inertia`` before each apply, so repeated
        notifications cannot accumulate. A regression that adds
        without first resetting would inflate the inertia by
        ``2*a*(axis ⊗ axis)`` after the second notify."""
        model = _build_anchored_revolute(mass=1.0, length=1.0, armature=0.0)
        solver = newton.solvers.SolverPhoenX(model, substeps=1)

        model.joint_armature.assign(np.array([1.5], dtype=np.float32))
        solver.notify_model_changed(int(SolverNotifyFlags.JOINT_DOF_PROPERTIES))
        inv_I_first = solver.bodies.inverse_inertia.numpy()[1].copy()

        # Notify again without changing armature.
        solver.notify_model_changed(int(SolverNotifyFlags.JOINT_DOF_PROPERTIES))
        inv_I_second = solver.bodies.inverse_inertia.numpy()[1]

        np.testing.assert_allclose(
            inv_I_second,
            inv_I_first,
            rtol=1.0e-5,
            err_msg=(
                "repeated notify_model_changed with the same armature must be "
                "idempotent; second call drifted from first by "
                f"{np.max(np.abs(inv_I_second - inv_I_first)):.6f}"
            ),
        )

    def test_revert_to_zero_armature_restores_inertia(self) -> None:
        """Setting ``armature == 0`` after a non-zero live update must
        restore the original (no-armature) body inertia bit-for-bit
        (modulo single-precision round-trip noise). Catches a
        regression that "remembers" armature once applied."""
        model = _build_anchored_revolute(mass=1.0, length=1.0, armature=0.0)
        solver = newton.solvers.SolverPhoenX(model, substeps=1)

        inv_I_zero_initial = solver.bodies.inverse_inertia.numpy()[1].copy()

        # Bump, then revert.
        model.joint_armature.assign(np.array([5.0], dtype=np.float32))
        solver.notify_model_changed(int(SolverNotifyFlags.JOINT_DOF_PROPERTIES))
        model.joint_armature.assign(np.array([0.0], dtype=np.float32))
        solver.notify_model_changed(int(SolverNotifyFlags.JOINT_DOF_PROPERTIES))

        inv_I_zero_after = solver.bodies.inverse_inertia.numpy()[1]
        np.testing.assert_allclose(
            inv_I_zero_after,
            inv_I_zero_initial,
            rtol=1.0e-4,
            err_msg=(
                "reverting armature to 0 must restore the original inv-inertia; "
                f"residual {np.max(np.abs(inv_I_zero_after - inv_I_zero_initial)):.6f}"
            ),
        )


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
                "live re-bake would still report ~T_expected_0)"
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
                "that skips the live re-bake would yield ratio ~1.0."
            ),
        )


if __name__ == "__main__":
    unittest.main()
