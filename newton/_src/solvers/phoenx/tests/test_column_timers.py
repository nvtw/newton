# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""End-to-end tests for the opt-in per-column wall-clock accumulator.

When :attr:`PhoenXWorld.enable_column_timers` is ``True``, every PGS
dispatch brackets its work with two ``%globaltimer`` reads and atomic-
adds the elapsed microseconds into the column's ``time_us`` slot. The
slot is zeroed at the start of every :meth:`PhoenXWorld.step`, so each
``step_report()`` returns the wall-clock cost of the *last* step.

CUDA-only -- inline PTX has no CPU fallback.
"""

from __future__ import annotations

import unittest

import numpy as np
import warp as wp

import newton
from newton._src.solvers.phoenx.constraints.constraint_joint import (
    JOINT_CONSTRAINT_TIME_US_OFFSET,
)
from newton._src.solvers.phoenx.tests.test_multi_world import _build_direct_pendulums, _make_direct_solver


def _build_timer_world(num_worlds: int):
    """Build one reusable direct-pendulum timer fixture."""
    model, _ = _build_direct_pendulums(world_count=num_worlds)
    solver = _make_direct_solver(model)
    solver._direct_equality_system = None
    solver.world._direct_equality_system = None
    solver.world.set_joint_pgs_ownership(np.ones(solver.world.num_joints, dtype=np.int32))
    state = model.state()
    newton.eval_fk(model, model.joint_q, model.joint_qd, state)
    control = model.control()

    def run_frames(frames: int) -> None:
        for _ in range(frames):
            state.clear_forces()
            solver.step(state, state, control, None, 1.0 / 60.0)

    return solver.world, run_frames


@unittest.skipUnless(wp.is_cuda_available(), "PhoenX column timers require CUDA")
class TestPhoenXColumnTimers(unittest.TestCase):
    def test_timer_modes_and_per_step_clearing(self) -> None:
        """Verify disabled, enabled, and per-step-cleared timer modes."""
        world, run_frames = _build_timer_world(num_worlds=4)

        run_frames(4)
        timer_row = world.constraints.data.numpy()[int(JOINT_CONSTRAINT_TIME_US_OFFSET), : world.num_joints]
        self.assertTrue(
            (timer_row == 0.0).all(),
            f"time_us row leaked while disabled: {timer_row.tolist()}",
        )
        report = world.step_report()
        self.assertIsNone(report.time_us_total_joints)
        self.assertIsNone(report.time_us_total_contacts)

        world.enable_column_timers = True
        run_frames(2)
        report = world.step_report()
        self.assertIsNotNone(report.time_us_total_joints)
        self.assertGreater(report.time_us_total_joints, 0.0)
        self.assertEqual(report.time_us_total_contacts, 0.0)
        self.assertEqual(report.time_us_total_cloth_triangles, 0.0)
        self.assertEqual(report.time_us_total_cloth_bending, 0.0)
        self.assertEqual(report.time_us_total_soft_tetrahedra, 0.0)

        first = report.time_us_total_joints
        run_frames(1)
        second = world.step_report().time_us_total_joints
        self.assertLess(second, 3.0 * first + 50.0)


if __name__ == "__main__":
    unittest.main()
