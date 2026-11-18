# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Bound activations refactor only the mechanisms whose drive bound changed."""

from __future__ import annotations

import unittest

import numpy as np
import warp as wp

import newton
from newton._src.solvers.phoenx.articulations.fixed_pattern_llt import FixedPatternPanelLLT

_INERTIA = ((0.01, 0.0, 0.0), (0.0, 0.01, 0.0), (0.0, 0.0, 0.01))


def _cuda_with_graph_capture() -> bool:
    device = wp.get_preferred_device()
    return bool(device.is_cuda and wp.is_mempool_enabled(device))


def _add_star(builder: newton.ModelBuilder, x: float, children: int, effort_limit: float) -> int:
    """Add a driven hub with revolute children; its factor has wide panels."""
    hub = builder.add_link(xform=wp.transform((x, 0.0, 1.0), wp.quat_identity()), mass=1.0, inertia=_INERTIA)
    hub_joint = builder.add_joint_revolute(
        parent=-1,
        child=hub,
        axis=(0.0, 1.0, 0.0),
        parent_xform=wp.transform((x, 0.0, 1.0), wp.quat_identity()),
        child_xform=wp.transform_identity(),
        target_ke=400.0,
        target_kd=4.0,
        effort_limit=effort_limit,
    )
    joints = [hub_joint]
    for k in range(children):
        angle = 2.0 * np.pi * k / children
        offset = (0.2 * np.cos(angle), 0.2 * np.sin(angle), 0.0)
        child = builder.add_link(
            xform=wp.transform((x + offset[0], offset[1], 1.0), wp.quat_identity()), mass=0.5, inertia=_INERTIA
        )
        joints.append(
            builder.add_joint_revolute(
                parent=hub,
                child=child,
                axis=(0.0, 0.0, 1.0),
                parent_xform=wp.transform(offset, wp.quat_identity()),
                child_xform=wp.transform_identity(),
            )
        )
    builder.add_articulation(joints)
    return hub_joint


def _simulate(*, partial: bool, steps: int = 12) -> tuple[np.ndarray, np.ndarray, newton.solvers.SolverPhoenX]:
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, -9.81), up_axis=newton.Axis.Z)
    # Enough independent wide mechanisms for the persistent ready-queue factor;
    # only the first one has a finite effort bound.
    driven = [_add_star(builder, 2.0 * index, 10, 1.0 if index == 0 else 0.0) for index in range(40)]
    model = builder.finalize()
    solver = newton.solvers.SolverPhoenX(
        model,
        substeps=1,
        solver_iterations=4,
        velocity_iterations=0,
        joint_mode="maximal_direct",
        step_layout="single_world",
    )
    system = solver._direct_equality_system
    if not partial:
        # Reference: every bound activation refactors all mechanisms.
        system._bound_row_mechanism = wp.empty(0, dtype=wp.int32, device=model.device)
    state = model.state()
    control = model.control()
    targets = control.joint_target_q.numpy()
    # A far target saturates the bounded hub drive after the first sweep.
    targets[model.joint_q_start.numpy()[driven[0]]] = 1.5
    control.joint_target_q.assign(targets)

    def step() -> None:
        state.clear_forces()
        solver.step(state, state, control, None, 1.0 / 120.0)

    step()
    with wp.ScopedCapture(device=model.device) as capture:
        step()
    for _ in range(steps):
        wp.capture_launch(capture.graph)
    return state.body_q.numpy(), state.body_qd.numpy(), solver


@unittest.skipUnless(_cuda_with_graph_capture(), "Partial direct refactor requires CUDA graph capture")
class TestDirectPartialRefactor(unittest.TestCase):
    def test_cooperative_fleet_refactors_only_marked_mechanisms(self) -> None:
        device = wp.get_preferred_device()
        if device.sm_count < 2:
            self.skipTest("Requires at least two CUDA SMs")
        count = (device.sm_count + 1) // 2
        # Dense three-panel mechanisms, with a partial final panel.
        rows = 35
        factor = FixedPatternPanelLLT(
            (rows,) * count,
            tuple(range(0, rows * count + 1, rows)),
            np.tile(np.arange(rows, dtype=np.int32), count),
            tuple(frozenset((body,)) for body in range(count) for _ in range(rows)),
            device=device,
        )
        self.assertEqual(factor.cooperative_factor_mechanism.size, count)
        self.assertTrue(factor.supports_partial_factor)
        matrix = np.zeros(factor.matrix.shape, dtype=np.float32)
        diagonal = factor.symbolic.diagonal_storage
        matrix[diagonal] = 4.0
        factor.matrix.assign(matrix)
        factor.compute()
        reference = factor.factor.numpy()
        # Change every input. Only the first mechanism may write its factor.
        matrix[diagonal] = 9.0
        factor.matrix.assign(matrix)
        mask = np.zeros(count, dtype=np.int32)
        mask[0] = 1
        factor.mechanism_dirty.assign(mask)
        factor.compute()
        actual = factor.factor.numpy()
        active_diagonal = diagonal[:rows]
        np.testing.assert_allclose(actual[active_diagonal], 3.0, rtol=2.0e-7)
        reference[active_diagonal] = actual[active_diagonal]
        np.testing.assert_array_equal(actual, reference)

    def test_saturating_drive_matches_full_refactor(self) -> None:
        q_partial, qd_partial, solver = _simulate(partial=True)
        system = solver._direct_equality_system
        self.assertTrue(system.solver.supports_partial_factor)
        self.assertGreater(system._bound_row_mechanism.shape[0], 0)
        self.assertTrue(bool(np.any(system.drive_saturated.numpy())), "the bounded drive must saturate")
        q_full, qd_full, _ = _simulate(partial=False)
        np.testing.assert_array_equal(q_partial, q_full)
        np.testing.assert_array_equal(qd_partial, qd_full)


if __name__ == "__main__":
    unittest.main()
