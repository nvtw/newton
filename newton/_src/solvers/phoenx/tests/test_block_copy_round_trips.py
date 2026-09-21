# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Keep copy synchronization only around actual global joint corrections."""

import unittest
from types import SimpleNamespace
from unittest.mock import Mock

from newton._src.solvers.phoenx.dispatch.single_world_mass_splitting import SingleWorldMassSplittingDispatcher


def make_world(requires_projection=False, solver_iterations=1, direct_joint_projection_passes=1):
    direct = SimpleNamespace(
        enabled=True,
        requires_global_projection=requires_projection,
        supports_async_factor=False,
        has_bounded_drives=False,
        prepare_matrix=Mock(),
        factor_async=Mock(),
        wait_factor=Mock(),
        prepare_and_factor=Mock(),
        solve=Mock(),
        resolve_bounded_drives=Mock(),
    )
    world = SimpleNamespace(
        _constraint_capacity=1,
        _direct_equality_system=direct,
        _regular_pgs_active_this_step=True,
        _combine_direct_prepare_projection=True,
        _direct_contact_response=None,
        _maximal_tree_projector=None,
        _reduced_constraints_active_this_step=False,
        substep_dt=0.01,
        solver_iterations=solver_iterations,
        direct_joint_projection_passes=direct_joint_projection_passes,
        _direct_joint_projection_iterations=frozenset(
            block * solver_iterations // direct_joint_projection_passes - 1
            for block in range(1, direct_joint_projection_passes + 1)
        ),
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
        "_wait_direct_factor",
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

    def test_direct_factor_overlaps_contact_preparation(self):
        """Launch an independent D6 factor before contact preparation and wait once."""
        world = make_world(requires_projection=True)
        direct = world._direct_equality_system
        direct.supports_async_factor = True
        world._color_group_data = object()
        SingleWorldMassSplittingDispatcher(world).solve(100.0)
        direct.prepare_and_factor.assert_not_called()
        direct.prepare_matrix.assert_called_once_with(100.0)
        direct.factor_async.assert_called_once_with()
        direct.wait_factor.assert_called_once_with()

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
                self.assertEqual(world._direct_equality_system.solve.call_count, 2)

    def test_global_projection_runs_after_final_iteration_only(self):
        """Keep mass-copy PGS iterations contiguous before exact projection."""
        world = make_world(True, solver_iterations=3)
        SingleWorldMassSplittingDispatcher(world).solve(100.0)
        self.assertEqual(world._direct_equality_system.solve.call_count, 2)
        self.assertEqual(world._mass_splitting_writeback.call_count, 2)
        self.assertEqual(world._mass_splitting_broadcast.call_count, 3)

    def test_direct_projection_passes_temporally_block_contact_sweeps(self):
        """Use a lean intermediate solve and fully refine the final projection."""
        world = make_world(True, solver_iterations=4, direct_joint_projection_passes=2)
        SingleWorldMassSplittingDispatcher(world).solve(100.0)
        calls = world._direct_equality_system.solve.call_args_list
        self.assertEqual([call.kwargs["use_bias"] for call in calls], [False, False, True])
        self.assertEqual(
            [call.kwargs.get("refine", True) for call in calls],
            [True, False, True],
        )
        self.assertEqual(world._mass_splitting_writeback.call_count, 3)
        self.assertEqual(world._mass_splitting_broadcast.call_count, 4)

    def test_solver_iterations_alternate_color_order(self):
        """Remove persistent directional bias without adding solver sweeps."""
        world = make_world(solver_iterations=4)
        SingleWorldMassSplittingDispatcher(world).solve(100.0)
        iteration_calls = world._singleworld_head_plus_tail_sweep.call_args_list[1:]
        self.assertEqual(
            [call.kwargs["reverse_colors"] for call in iteration_calls],
            [False, True, False, True],
        )

    def test_global_corrections_retain_copy_synchronization(self):
        """Preserve direct and finite-drive correction round trips."""
        world = make_world(True)
        SingleWorldMassSplittingDispatcher(world).solve(100.0)
        self.assertEqual(world._mass_splitting_broadcast.call_count, 3)
        self.assertEqual(world._direct_equality_system.solve.call_count, 2)
        self.assertEqual(world._direct_equality_system.resolve_bounded_drives.call_count, 2)


if __name__ == "__main__":
    unittest.main()
