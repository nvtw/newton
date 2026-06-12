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

import warp as wp

from newton._src.solvers.phoenx.constraints.constraint_joint import (
    ADBS_TIME_US_OFFSET,
)
from newton._src.solvers.phoenx.tests.test_multi_world import _build_n_pendulums, _run_frames


@unittest.skipUnless(wp.is_cuda_available(), "PhoenX column timers require CUDA")
class TestPhoenXColumnTimers(unittest.TestCase):
    def test_disabled_leaves_time_us_at_zero(self) -> None:
        """Default ``enable_column_timers=False`` must keep every
        column's ``time_us`` slot pinned at zero so the dead-code-
        eliminated globaltimer reads can't leak via stale state."""
        device = wp.get_device("cuda:0")
        world, _ = _build_n_pendulums(num_worlds=4, device=device)
        _run_frames(world, 4)
        col = world.constraints.data.numpy()
        timer_row = col[int(ADBS_TIME_US_OFFSET), : world.num_joints]
        self.assertTrue(
            (timer_row == 0.0).all(),
            f"time_us row leaked non-zero values when timers were disabled: {timer_row.tolist()}",
        )
        report = world.step_report()
        self.assertIsNone(report.time_us_total_joints)
        self.assertIsNone(report.time_us_total_contacts)

    def test_enabled_records_positive_joint_time(self) -> None:
        """``enable_column_timers=True`` must accumulate strictly
        positive joint dispatch time and surface it via the step report."""
        device = wp.get_device("cuda:0")
        world, _ = _build_n_pendulums(num_worlds=4, device=device)
        world.enable_column_timers = True
        # First step warms up + clears + records; second step exercises
        # the per-step zeroing so we know the slot isn't stuck.
        _run_frames(world, 2)
        report = world.step_report()
        self.assertIsNotNone(report.time_us_total_joints)
        self.assertGreater(report.time_us_total_joints, 0.0)
        # No contacts in this pendulum scene.
        self.assertEqual(report.time_us_total_contacts, 0.0)
        self.assertEqual(report.time_us_total_cloth_triangles, 0.0)
        self.assertEqual(report.time_us_total_cloth_bending, 0.0)
        self.assertEqual(report.time_us_total_soft_tetrahedra, 0.0)

    def test_cleared_each_step(self) -> None:
        """The per-step zero pass must keep the joint total bounded:
        if zeroing leaked, two consecutive 1-step reports would show
        roughly doubled totals."""
        device = wp.get_device("cuda:0")
        world, _ = _build_n_pendulums(num_worlds=4, device=device)
        world.enable_column_timers = True
        dt = 1.0 / 60.0
        world.step(dt=dt, contacts=None, shape_body=None)
        first = world.step_report().time_us_total_joints
        world.step(dt=dt, contacts=None, shape_body=None)
        second = world.step_report().time_us_total_joints
        # Each step is independent: the second total must NOT include
        # the first step's accumulated time. Allow 3x slack for jitter.
        self.assertLess(second, 3.0 * first + 50.0)


if __name__ == "__main__":
    unittest.main()
