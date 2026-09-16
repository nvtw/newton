# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Compare final-substep relaxation with an independently scheduled reference."""

import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import warp as wp

import newton
from newton._src.solvers.phoenx.articulations.reduced import ReducedPhoenXArticulation
from newton._src.solvers.phoenx.dispatch.multi_world import MultiWorldDispatcher
from newton._src.solvers.phoenx.dispatch.single_world import SingleWorldDispatcher
from newton._src.solvers.phoenx.solver_phoenx import PhoenXWorld
from newton._src.solvers.phoenx.tests.test_rigid_split_prepare import run_scene


class TestReducedRelaxationSchedule(unittest.TestCase):
    def test_dispatchers_skip_all_impulses_without_relaxation(self):
        """Skip tree projection and reduced solves when relaxation is disabled."""
        for dispatcher in (SingleWorldDispatcher, MultiWorldDispatcher):
            with self.subTest(dispatcher=dispatcher.__name__):
                world = SimpleNamespace(
                    _active_velocity_iterations=0,
                    _constraint_capacity=1,
                    _regular_pgs_active_this_step=False,
                    _direct_equality_system=None,
                    _maximal_tree_projector=Mock(),
                    _direct_tree_contacts=False,
                    _reduced_constraints_active_this_step=True,
                    _reduced_articulation=Mock(),
                    substep_dt=1.0 / 120.0,
                )
                dispatcher(world).relax(120.0)
                world._maximal_tree_projector.project.assert_not_called()
                world._reduced_articulation.solve_constraints.assert_not_called()

    def test_reduced_solve_obeys_temporal_schedule(self):
        """Skip reduced impulses and preparation before final relaxation."""
        world = PhoenXWorld.__new__(PhoenXWorld)
        world.substeps = 3
        world.velocity_iterations = 2
        world.solver_iterations = 4
        world.sor_boost = 1.0
        world.max_contact_columns = 0
        world._reduced_contacts_active_this_step = False
        articulation = SimpleNamespace(
            _kinematics_current=True,
            system=Mock(),
            state=object(),
            control=object(),
            model=object(),
            bodies=object(),
            loop_system=Mock(),
            _refresh_impulse_response=Mock(),
        )
        for policy in ("final_substep", "each_substep"):
            with self.subTest(policy=policy):
                world.velocity_relaxation = policy
                articulation.system.reset_mock()
                articulation.loop_system.reset_mock()
                for substep in range(world.substeps):
                    world._current_substep_index = substep
                    ReducedPhoenXArticulation.solve_constraints(articulation, world, 120.0, relax=True)
                    expected = substep + 1 if policy == "each_substep" else int(substep == 2)
                    self.assertEqual(articulation.loop_system.solve.call_count, expected)
                    self.assertEqual(articulation.system.factor.call_count, expected)
                for call in articulation.loop_system.solve.call_args_list:
                    self.assertEqual(call.args[4], 2)
                    self.assertFalse(call.kwargs["use_bias"])
                    self.assertFalse(call.kwargs["warmstart"])
        world.velocity_relaxation = "final_substep"
        world._current_substep_index = 0
        articulation.loop_system.reset_mock()
        ReducedPhoenXArticulation.solve_constraints(articulation, world, 120.0, relax=False)
        self.assertEqual(articulation.loop_system.solve.call_args.args[4], 4)
        self.assertTrue(articulation.loop_system.solve.call_args.kwargs["use_bias"])


@unittest.skipUnless(wp.is_cuda_available(), "Relaxation scheduling requires CUDA")
class TestVelocityRelaxationSchedule(unittest.TestCase):
    def test_final_relaxation_matches_reference_with_changing_contacts(self):
        """Match explicit scheduling while contact ownership changes."""
        original_init = newton.solvers.SolverPhoenX.__init__
        original_forces = PhoenXWorld._integrate_forces_and_gravity

        def run(final_policy, mass_splitting):
            def initialize(solver, *args, **kwargs):
                kwargs.update(
                    substeps=3,
                    velocity_iterations=1,
                    velocity_relaxation="final_substep" if final_policy else "each_substep",
                )
                original_init(solver, *args, **kwargs)

            def reference_forces(world):
                world.velocity_iterations = int(world._current_substep_index == 2)
                return original_forces(world)

            with patch.object(newton.solvers.SolverPhoenX, "__init__", initialize):
                if final_policy:
                    return run_scene(True, mass_splitting=mass_splitting)
                with patch.object(PhoenXWorld, "_integrate_forces_and_gravity", reference_forces):
                    return run_scene(True, mass_splitting=mass_splitting)

        for split in (False, True):
            with self.subTest(mass_splitting=split):
                reference = run(False, split)
                candidate = run(True, split)
                for before, after in zip(reference, candidate, strict=True):
                    for key in before:
                        np.testing.assert_array_equal(before[key], after[key], err_msg=key)

    def test_invalid_policy_is_rejected(self):
        """Reject an unknown relaxation schedule."""
        builder = newton.ModelBuilder()
        model = builder.finalize(device="cuda:0")
        with self.assertRaisesRegex(ValueError, "velocity_relaxation"):
            newton.solvers.SolverPhoenX(model, velocity_relaxation="sometimes")


if __name__ == "__main__":
    unittest.main()
