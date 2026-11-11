# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

import inspect
import unittest
from types import SimpleNamespace
from unittest.mock import Mock

import warp as wp

import newton
from newton._src.solvers.phoenx.dispatch.multi_world import MultiWorldDispatcher
from newton._src.solvers.phoenx.examples.example_humanoid import Example as HumanoidExample
from newton._src.solvers.phoenx.simulation import PhoenXWorld
from newton._src.solvers.phoenx.solver import (
    _can_combine_direct_prepare_projection,
    _estimate_contact_column_max_phoenx,
    _resolve_auto_step_layout,
)


class TestPhoenXAutoPerformancePolicy(unittest.TestCase):
    def test_solver_exposes_one_explicit_joint_mode(self):
        parameters = inspect.signature(newton.solvers.SolverPhoenX).parameters
        self.assertEqual(parameters["joint_mode"].default, "maximal_direct")
        self.assertNotIn("articulation_mode", parameters)
        self.assertNotIn("joint_solver", parameters)

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

    def test_large_reduced_world_selects_single_world_by_free_bodies(self):
        """One world of reduced robots among many free bodies solves its contacts color by color."""
        reduced = {"articulation_mode": "reduced", "has_joints": True, "body_count": 200000}
        self.assertEqual(self._resolve(**reduced, free_body_count=32768), "single_world")
        self.assertEqual(self._resolve(**reduced, free_body_count=32767), "multi_world")
        self.assertEqual(self._resolve(**reduced, free_body_count=32768, num_worlds=2), "multi_world")

    def test_free_body_count_ignores_jointed_bodies(self):
        """Bodies alone on a free joint or without a joint count as free; hinged ones do not."""
        import newton  # noqa: PLC0415
        from newton._src.solvers.phoenx.solver import _free_body_count  # noqa: PLC0415

        builder = newton.ModelBuilder()
        builder.add_body()  # alone on a free joint, in its own articulation
        base = builder.add_link()
        arm = builder.add_link()
        builder.add_joint_free(base)
        builder.add_joint_revolute(base, arm)
        builder.add_articulation([builder.joint_count - 2, builder.joint_count - 1])
        model = builder.finalize(device="cpu")
        self.assertEqual(_free_body_count(model), 1)

        # A model stripped of its articulation data: every free joint then stands alone.
        model.joint_articulation = None
        self.assertEqual(_free_body_count(model), 2)

    def test_contact_columns_are_bounded_by_the_candidate_pairs(self):
        """One large world sizes its columns by the broad phase's pair budget, not by the contacts."""
        from newton._src.solvers.phoenx.solver import _estimate_contact_column_max_phoenx  # noqa: PLC0415

        # 4096 warehouse robots: 19.9M contact slots overflowed the 2^24 limit of the partitioner.
        dense = SimpleNamespace(
            shape_contact_pair_count=40_000_000, _collision_pipeline=SimpleNamespace(shape_pairs_max=8_271_072)
        )
        self.assertEqual(_estimate_contact_column_max_phoenx(dense, 19_896_785), 8_271_072)
        listed = SimpleNamespace(
            shape_contact_pair_count=5000, _collision_pipeline=SimpleNamespace(shape_pairs_max=5000)
        )
        self.assertEqual(_estimate_contact_column_max_phoenx(listed, 50_000), 5000)
        unbounded = SimpleNamespace(shape_contact_pair_count=5000)
        self.assertEqual(_estimate_contact_column_max_phoenx(unbounded, 3000), 3000)

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
            supports_async_factor=False,
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
            _warm_start_owned_contacts=lambda: events.append("warm"),
            _maximal_tree_projector=None,
            _reduced_constraints_active_this_step=False,
        )
        MultiWorldDispatcher(world).solve(wp.float32(1.0))
        self.assertEqual(events[:5], ["factor", "prepare", "warm", ("solve", False), ("bounds", False)])

    def test_direct_relax_always_refreshes_integrated_geometry(self) -> None:
        """Relaxation re-prepares the direct system even for one unconstrained sweep.

        Integration moves centers of mass, so reusing the biased solve's levers would apply
        paired impulses at stale world points and create angular momentum.
        """
        direct = SimpleNamespace(
            enabled=True,
            has_bounded_drives=False,
            supports_async_factor=True,
            refresh_geometry=Mock(),
            prepare_matrix=Mock(),
            factor_async=Mock(),
            prepare_and_factor=Mock(),
        )
        world = SimpleNamespace(
            _active_velocity_iterations=1,
            _contact_input_active_this_step=False,
            num_particles=0,
            _contact_patch_enabled=False,
            _direct_equality_system=direct,
            mass_splitting_enabled=True,
            _regular_pgs_active_this_step=True,
            _combine_direct_prepare_projection=True,
            _direct_contact_response=None,
            _maximal_contact_response=None,
            _direct_factor_pending=False,
            _direct_tree_contacts=False,
            _current_substep_index=0,
            substeps=1,
            prepare_refresh_stride=1,
            _reduced_articulation=None,
        )
        PhoenXWorld._refresh_owned_relax_geometry(world, wp.float32(1.0))
        direct.refresh_geometry.assert_called_once()
        direct.factor_async.assert_called_once()

    def test_direct_contact_sweeps_share_one_launch(self) -> None:
        """Run all biased direct-contact iterations in one ordered sweep call."""
        response = SimpleNamespace(compute=Mock())
        schedule = SimpleNamespace(solve=Mock())
        world = SimpleNamespace(
            _contact_input_active_this_step=True,
            _direct_contact_response=response,
            _direct_contact_schedule=schedule,
            _direct_contact_response_current=False,
            _contact_container=object(),
            bodies=object(),
            solver_iterations=3,
            velocity_iterations=1,
            substep_dt=0.01,
            sor_boost=1.0,
        )
        PhoenXWorld._solve_direct_contacts(world, use_bias=True, refresh_mobility=True)
        schedule.solve.assert_called_once()
        self.assertEqual(schedule.solve.call_args.kwargs["iterations"], 3)
        self.assertTrue(schedule.solve.call_args.kwargs["use_bias"])
        response.compute.assert_called_once()

    def test_direct_contact_response_reused_after_warm_start(self) -> None:
        """Reuse the warm start's mobility for the biased sweep of the same substep."""
        response = SimpleNamespace(compute=Mock())
        schedule = SimpleNamespace(solve=Mock())
        world = SimpleNamespace(
            _contact_input_active_this_step=True,
            _direct_contact_response=response,
            _direct_contact_schedule=schedule,
            _direct_contact_response_current=True,
            _contact_container=object(),
            bodies=object(),
            solver_iterations=2,
            substep_dt=0.01,
            sor_boost=1.0,
        )
        PhoenXWorld._solve_direct_contacts(world, use_bias=True, refresh_mobility=True)
        response.compute.assert_not_called()
        self.assertFalse(world._direct_contact_response_current)
        PhoenXWorld._solve_direct_contacts(world, use_bias=True, refresh_mobility=True)
        response.compute.assert_called_once()

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
