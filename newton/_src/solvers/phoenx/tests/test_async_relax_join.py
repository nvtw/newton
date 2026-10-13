# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Check ordering of asynchronous joint factors and contact relaxation."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

from newton._src.solvers.phoenx.dispatch import single_world_mass_splitting_unrolled as dispatch
from newton._src.solvers.phoenx.simulation import PhoenXWorld


class TestAsyncRelaxJoin(unittest.TestCase):
    def _run_relax(self, regular_contacts):
        events = []
        world = SimpleNamespace(
            max_colored_partitions=2,
            _constraint_capacity=8,
            _active_velocity_iterations=1,
            _singleworld_overflow_only_mass_splitting=True,
            _regular_pgs_active_this_step=regular_contacts,
            _direct_factor_pending=True,
            substep_dt=0.01,
            _contact_container_solve=None,
            _partitioner=SimpleNamespace(begin_sweep=lambda: events.append("begin_contacts")),
            _singleworld_kernels=lambda: (None, None, None, None, "relax_head", None),
            _launch_singleworld_head=lambda _kernel, _idt, _threshold, _cc, step: events.append(f"color_{step}"),
            _mass_splitting_average_overflow_into_bodies=lambda: events.append("average_overflow"),
        )

        def project(*, use_bias):
            self.assertFalse(use_bias)
            self.assertFalse(world._direct_factor_pending, "Joint projection consumed an unfinished factor")
            events.append("project_joints")

        world._direct_equality_system = SimpleNamespace(
            enabled=True,
            wait_factor=lambda: events.append("wait_factor"),
            solve=project,
            resolve_bounded_drives=lambda *_args, **_kwargs: None,
        )
        world._wait_direct_factor = lambda: PhoenXWorld._wait_direct_factor(world)
        with patch.object(dispatch, "solve_auxiliary_constraints"):
            dispatch.SingleWorldMassSplittingUnrolledDispatcher(world).relax(100.0)
        return events, world

    def test_join_follows_contact_relaxation(self):
        """Keep the factor branch independent until all contact updates are complete."""
        events, world = self._run_relax(True)
        self.assertEqual(
            events,
            ["begin_contacts", "color_0", "color_1", "color_2", "average_overflow", "wait_factor", "project_joints"],
        )
        self.assertFalse(world._direct_factor_pending)
        # The same factor remains available for the next substep's reuse.
        world._wait_direct_factor()
        self.assertEqual(events.count("wait_factor"), 1)

    def test_joint_only_relaxation_also_joins(self):
        """Wait before exact projection when there are no regular PGS constraints."""
        events, world = self._run_relax(False)
        self.assertEqual(events, ["wait_factor", "project_joints"])
        self.assertFalse(world._direct_factor_pending)


if __name__ == "__main__":
    unittest.main()
