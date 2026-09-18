# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Keep copy synchronization only around actual global joint corrections."""

import unittest
from types import SimpleNamespace
from unittest.mock import Mock

from newton._src.solvers.phoenx.dispatch.single_world_mass_splitting import SingleWorldMassSplittingDispatcher


def make_world(requires_projection=False, solver_iterations=1):
    direct = SimpleNamespace(
        enabled=True,
        requires_global_projection=requires_projection,
        prepare_and_factor=Mock(),
        solve=Mock(),
        resolve_bounded_drives=Mock(),
    )
    world = SimpleNamespace(
        _constraint_capacity=1,
        _direct_equality_system=direct,
        _regular_pgs_active_this_step=True,
        _direct_contact_response=None,
        _maximal_tree_projector=None,
        _reduced_constraints_active_this_step=False,
        substep_dt=0.01,
        solver_iterations=solver_iterations,
        joint_refinement_iterations=0,
        _colored_contact_rows=False,
        _color_group_data=None,
        _contact_container_solve=None,
        _partitioner=SimpleNamespace(begin_sweep=Mock()),
        _singleworld_kernels=Mock(return_value=(None,) * 6),
        _refresh_prepare_this_substep=Mock(return_value=True),
        _active_velocity_iterations=2,
    )
    for name in (
        "_mass_splitting_broadcast",
        "_mass_splitting_writeback",
        "_mass_splitting_average_and_broadcast",
        "_singleworld_head_plus_tail_sweep",
        "_warm_start_owned_contacts",
        "_solve_direct_contacts",
        "_run_cached_prepare_bookkeeping",
    ):
        setattr(world, name, Mock())
    return world


class TestBlockCopyRoundTrips(unittest.TestCase):
    def test_local_blocks_prepare_without_global_copy_round_trips(self):
        """Prepare local blocks but broadcast only once before the actual row sweep."""
        world = make_world()
        SingleWorldMassSplittingDispatcher(world).solve(100.0)
        world._direct_equality_system.prepare_and_factor.assert_called_once_with(100.0)
        self.assertEqual(world._mass_splitting_broadcast.call_count, 1)
        self.assertEqual(world._mass_splitting_writeback.call_count, 1)
        world._direct_equality_system.solve.assert_not_called()

    def test_owned_contact_paths_keep_original_synchronization(self):
        """Retain round trips whenever an articulated contact path is active."""
        for path in ("direct", "maximal", "reduced"):
            with self.subTest(path=path):
                world = make_world()
                if path == "direct":
                    world._direct_contact_response = object()
                elif path == "maximal":
                    world._maximal_tree_projector = SimpleNamespace(project=Mock())
                    world._solve_maximal_articulated_contacts = Mock()
                else:
                    world._reduced_constraints_active_this_step = True
                    world._reduced_articulation = SimpleNamespace(solve_constraints=Mock())
                SingleWorldMassSplittingDispatcher(world).solve(100.0)
                self.assertEqual(world._mass_splitting_broadcast.call_count, 3)
                self.assertEqual(world._direct_equality_system.solve.call_count, 3)

    def test_global_projection_runs_after_final_iteration_only(self):
        """Keep mass-copy PGS iterations contiguous before exact projection."""
        world = make_world(True, solver_iterations=3)
        SingleWorldMassSplittingDispatcher(world).solve(100.0)
        self.assertEqual(world._direct_equality_system.solve.call_count, 3)
        self.assertEqual(world._mass_splitting_writeback.call_count, 2)
        self.assertEqual(world._mass_splitting_broadcast.call_count, 3)

    def test_global_corrections_retain_copy_synchronization(self):
        """Preserve direct and finite-drive correction round trips."""
        world = make_world(True)
        SingleWorldMassSplittingDispatcher(world).solve(100.0)
        self.assertEqual(world._mass_splitting_broadcast.call_count, 3)
        self.assertEqual(world._direct_equality_system.solve.call_count, 3)
        self.assertEqual(world._direct_equality_system.resolve_bounded_drives.call_count, 2)


if __name__ == "__main__":
    unittest.main()
