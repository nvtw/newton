# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""The device activity flag of exact joint solves matches the conditional node."""

from __future__ import annotations

import unittest
from unittest.mock import PropertyMock, patch

import numpy as np
import warp as wp

import newton
from newton._src.solvers.phoenx.articulations.fixed_pattern_llt import FixedPatternPanelLLT
from newton._src.solvers.phoenx.tests.test_direct_partial_refactor import _add_star


def _cuda_with_graph_capture() -> bool:
    device = wp.get_preferred_device()
    return bool(device.is_cuda and wp.is_mempool_enabled(device))


def _build(gravity: float) -> newton.Model:
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, gravity), up_axis=newton.Axis.Z)
    # Wide hub mechanisms use the cooperative mechanism solve, not the fused
    # per-subgroup projection, so the solve keeps its activity flag.
    for index in range(40):
        _add_star(builder, 2.0 * index, 10, 0.0)
    return builder.finalize()


def _simulate(gravity: float, *, gated: bool, steps: int = 10):
    model = _build(gravity)
    with patch.object(FixedPatternPanelLLT, "supports_active_flag", new_callable=PropertyMock) as flag:
        flag.return_value = gated
        solver = newton.solvers.SolverPhoenX(
            model,
            substeps=1,
            solver_iterations=2,
            velocity_iterations=1,
            joint_mode="maximal_direct",
            step_layout="single_world",
        )
        state = model.state()
        control = model.control()

        def step() -> None:
            state.clear_forces()
            solver.step(state, state, control, None, 1.0 / 120.0)

        step()
        with wp.ScopedCapture(device=model.device) as capture:
            step()
    for _ in range(steps):
        wp.capture_launch(capture.graph)
    return state.body_q.numpy(), state.body_qd.numpy(), solver


@unittest.skipUnless(_cuda_with_graph_capture(), "The joint solve activity flag requires CUDA graph capture")
class TestDirectSolveActiveFlag(unittest.TestCase):
    def test_flag_matches_conditional_node(self) -> None:
        """Both active (falling) and inactive (resting) solves match bitwise."""
        for gravity in (-9.81, 0.0):
            with self.subTest(gravity=gravity):
                q_gated, qd_gated, solver = _simulate(gravity, gated=True)
                q_node, qd_node, _ = _simulate(gravity, gated=False)
                np.testing.assert_array_equal(q_gated, q_node)
                np.testing.assert_array_equal(qd_gated, qd_node)
                system = solver._direct_equality_system
                self.assertTrue(type(system.solver).supports_active_flag.fget(system.solver))
                self.assertIsNone(system._fused_projection, "the flag replaces the conditional only without fusion")
                if gravity == 0.0:
                    # The last unbiased solve of resting mechanisms has no work.
                    self.assertEqual(int(system.solve_active.numpy()[0]), 0)


if __name__ == "__main__":
    unittest.main()
