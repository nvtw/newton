# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Check that each unrolled color launch has an immutable sweep index."""

import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import warp as wp

from newton._src.solvers.phoenx.dispatch.single_world_mass_splitting_unrolled import (
    SingleWorldMassSplittingUnrolledDispatcher,
)
from newton._src.solvers.phoenx.simulation import PhoenXWorld
from newton._src.solvers.phoenx.simulation_kernels import advance_singleworld_color_cursor_kernel


class TestUnrolledColorDispatch(unittest.TestCase):
    def test_each_head_launch_receives_its_fixed_step(self):
        launch = Mock()
        average = Mock()
        world = SimpleNamespace(
            max_colored_partitions=3,
            _singleworld_overflow_only_mass_splitting=True,
            _launch_singleworld_head=launch,
            _mass_splitting_average_overflow_into_bodies=average,
        )
        dispatcher = SingleWorldMassSplittingUnrolledDispatcher(world)
        kernel = object()

        dispatcher._unrolled_sweep(kernel, wp.float32(120.0))

        self.assertEqual([call.args[4] for call in launch.call_args_list], [0, 1, 2, 3])
        average.assert_called_once_with()

    def test_dynamic_head_advances_cursor_after_grid_launch(self):
        kernel = object()
        cursor = object()
        active = object()
        partitioner = SimpleNamespace(
            element_ids_by_color=object(),
            color_starts=object(),
            color_family_starts=object(),
            num_colors=object(),
            color_cursor=cursor,
            sweep_direction=object(),
        )
        world = SimpleNamespace(
            _active_contact_views=Mock(return_value=object()),
            _contact_container=object(),
            max_colored_partitions=None,
            mass_splitting_batch_size=1,
            sor_boost=1.0,
            constraints=object(),
            _contact_cols=object(),
            _colored_contact_headers=False,
            bodies=object(),
            _particles_or_sentinel=Mock(return_value=object()),
            _partitioner=partitioner,
            num_joints=0,
            _joint_pgs_enabled=object(),
            num_cloth_triangles=0,
            num_cloth_bending=0,
            num_soft_tetrahedra=0,
            num_soft_hexahedra=0,
            num_bodies=2,
            _singleworld_total_threads=256,
            _head_active=active,
            _copy_state=object(),
            device=object(),
        )

        with patch("newton._src.solvers.phoenx.simulation.wp.launch") as launch:
            PhoenXWorld._launch_singleworld_head(world, kernel, wp.float32(120.0), wp.int32(16))

        self.assertEqual(launch.call_count, 2)
        self.assertIs(launch.call_args_list[0].args[0], kernel)
        self.assertIs(launch.call_args_list[1].args[0], advance_singleworld_color_cursor_kernel)
        self.assertEqual(launch.call_args_list[1].kwargs["inputs"], [cursor, active])


if __name__ == "__main__":
    unittest.main()
