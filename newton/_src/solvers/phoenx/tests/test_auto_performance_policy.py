# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

import unittest
from types import SimpleNamespace

import warp as wp

import newton
from newton._src.solvers.phoenx.dispatch.multi_world import MultiWorldDispatcher
from newton._src.solvers.phoenx.examples.example_humanoid import Example as HumanoidExample
from newton._src.solvers.phoenx.solver import (
    _can_combine_direct_prepare_projection,
    _estimate_contact_column_max_phoenx,
    _resolve_auto_step_layout,
)


class TestPhoenXAutoPerformancePolicy(unittest.TestCase):
    def _resolve(self, **overrides):
        options = {
            "step_layout": "auto",
            "num_worlds": 1,
            "body_count": 2048,
            "has_joints": False,
            "has_deformables": False,
            "has_shapes": True,
            "contact_friction_model": "point",
            "articulation_mode": "maximal",
        }
        options.update(overrides)
        return _resolve_auto_step_layout(**options)

    def test_large_rigid_contact_world_selects_single_world(self):
        self.assertEqual(self._resolve(), "single_world")

    def test_noneligible_topologies_keep_multi_world(self):
        for overrides in (
            {"body_count": 2047},
            {"num_worlds": 2},
            {"has_joints": True},
            {"has_deformables": True},
        ):
            with self.subTest(**overrides):
                self.assertEqual(self._resolve(**overrides), "multi_world")

    def test_combined_projection_requires_velocity_invariant_prepare(self) -> None:
        """Combine direct projection only when row preparation ignores velocity."""
        self.assertTrue(_can_combine_direct_prepare_projection(False, "point", "maximal"))
        self.assertFalse(_can_combine_direct_prepare_projection(True, "point", "maximal"))
        self.assertFalse(_can_combine_direct_prepare_projection(False, "patch", "maximal"))
        self.assertFalse(_can_combine_direct_prepare_projection(False, "point", "reduced"))

    def test_combined_projection_follows_prepare_warm_start(self) -> None:
        """Project direct equalities once after velocity-invariant preparation."""
        events = []
        direct = SimpleNamespace(
            enabled=True,
            prepare_and_factor=lambda _idt: events.append("factor"),
            solve=lambda *, use_bias: events.append(("solve", use_bias)),
            resolve_bounded_drives=lambda _idt, *, use_bias: events.append(("bounds", use_bias)),
        )
        world = SimpleNamespace(
            _direct_equality_system=direct,
            _regular_pgs_active_this_step=True,
            _multi_world_scheduler="block_world",
            _combine_direct_prepare_projection=True,
            _block_world_supported=lambda: True,
            _solve_main_block_world=lambda **_kwargs: events.append("prepare"),
            solver_iterations=0,
            _solve_direct_contacts=lambda **_kwargs: None,
            _maximal_tree_projector=None,
            _reduced_constraints_active_this_step=False,
        )
        MultiWorldDispatcher(world).solve(wp.float32(1.0))
        self.assertEqual(events[:4], ["factor", "prepare", ("solve", False), ("bounds", False)])

    def test_explicit_overrides_are_preserved(self):
        self.assertEqual(self._resolve(step_layout="multi_world"), "multi_world")
        self.assertEqual(self._resolve(step_layout="single_world"), "single_world")

    def test_contact_columns_use_shape_pair_capacity(self) -> None:
        """Size contact columns from shape pairs instead of contact points."""
        model = SimpleNamespace(shape_contact_pair_count=38_000)
        self.assertEqual(_estimate_contact_column_max_phoenx(model, 190_000), 38_000)
        self.assertEqual(_estimate_contact_column_max_phoenx(model, 20_000), 20_000)
        self.assertEqual(
            _estimate_contact_column_max_phoenx(SimpleNamespace(shape_contact_pair_count=4), 50_000), 1_000
        )
        self.assertEqual(_estimate_contact_column_max_phoenx(SimpleNamespace(), 50_000), 50_000)

    def test_humanoid_exposes_coordinate_mode_switch(self) -> None:
        """Select both full and reduced humanoid coordinate modes globally."""
        parser = HumanoidExample.create_parser()
        self.assertFalse(parser.parse_args([]).reduced_coordinates)
        self.assertTrue(parser.parse_args(["--reduced-coordinates"]).reduced_coordinates)
        self.assertFalse(parser.parse_args(["--no-reduced-coordinates"]).reduced_coordinates)

    def test_removed_simple_solver_points_to_mini(self) -> None:
        """Deprecate the old flavor argument and direct experiments to Mini."""
        model = newton.ModelBuilder().finalize()
        with self.assertWarns(DeprecationWarning), self.assertRaisesRegex(ValueError, "PhoenX Mini"):
            newton.solvers.SolverPhoenX(model, solver_flavor="simple")


if __name__ == "__main__":
    unittest.main()
