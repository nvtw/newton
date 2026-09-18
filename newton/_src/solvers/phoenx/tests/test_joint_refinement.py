# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Joint refinement uses the existing paired impulses and copy reconciliation."""

import itertools
import unittest

import numpy as np
import warp as wp

import newton
from newton._src.solvers.phoenx.tests.test_contact_chunks import energy
from newton._src.solvers.phoenx.tests.test_contact_coupling import _total_momentum
from newton._src.solvers.phoenx.tests.test_direct_drive import _cuda_with_graph_capture


@unittest.skipUnless(_cuda_with_graph_capture(), "Grouped joint refinement requires CUDA")
class TestJointRefinement(unittest.TestCase):
    def test_captured_refinement_matches_joint_sweeps_and_conserves_momentum(self):
        builder = newton.ModelBuilder(gravity=wp.vec3(0.0))
        bodies = [
            builder.add_body(mass=m, inertia=wp.mat33(m, 0.0, 0.0, 0.0, m, 0.0, 0.0, 0.0, m))
            for m in (1.0, 2.0, 3.0, 4.0)
        ]
        for parent, child in itertools.pairwise(bodies):
            builder.add_joint_fixed(parent=parent, child=child)
        model = builder.finalize(device="cuda:0")
        outputs = []
        for mixed, refinement in ((3, 0), (1, 2), (1, 0)):
            solver = newton.solvers.SolverPhoenX(
                model,
                articulation_mode="maximal",
                step_layout="single_world",
                joint_solver="block_pgs",
                mass_splitting=True,
                mass_splitting_color_group_size=1,
                solver_iterations=mixed,
                joint_refinement_iterations=refinement,
                velocity_iterations=1,
                sor_boost=1.0,
            )
            state = model.state()
            velocity = np.zeros((4, 6), dtype=np.float32)
            velocity[0] = (0.7, -0.4, 0.2, 0.3, -0.1, 0.2)
            state.body_qd.assign(velocity)
            momentum = _total_momentum(model, state)
            initial_energy = energy(model, state)
            with wp.ScopedCapture(device=model.device) as capture:
                state.clear_forces()
                solver.step(state, state, model.control(), None, 0.001)
            wp.capture_launch(capture.graph)
            np.testing.assert_allclose(_total_momentum(model, state), momentum, atol=2e-6, rtol=0)
            self.assertLessEqual(energy(model, state), initial_energy + 1e-6)
            self.assertGreater(int(solver.world._copy_state.count_per_node.numpy().max()), 1)
            outputs.append(state.body_qd.numpy())
        np.testing.assert_array_equal(outputs[0], outputs[1])
        self.assertGreater(float(np.max(np.abs(outputs[0] - outputs[2]))), 1e-4)

    def test_invalid_refinement_options_are_rejected(self):
        builder = newton.ModelBuilder()
        builder.add_body(mass=1.0, inertia=wp.mat33(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0))
        model = builder.finalize(device="cuda:0")
        for value in (-1, 1.5, True):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "nonnegative integer"):
                newton.solvers.SolverPhoenX(model, joint_refinement_iterations=value)
        for options in (
            {},
            {"mass_splitting": True},
            {"joint_solver": "block_pgs", "articulation_mode": "maximal", "step_layout": "single_world"},
        ):
            with self.subTest(options=options), self.assertRaisesRegex(ValueError, "joint_refinement_iterations"):
                newton.solvers.SolverPhoenX(model, joint_refinement_iterations=1, **options)


if __name__ == "__main__":
    unittest.main()
