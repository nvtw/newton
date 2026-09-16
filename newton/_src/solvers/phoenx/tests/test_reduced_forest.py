# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Physical response checks for the compact forest contact prototype."""

import unittest

import numpy as np
import warp as wp

import newton
from newton._src.solvers.phoenx.articulations.reduced import ReducedArticulationSystem
from newton._src.solvers.phoenx.articulations.reduced_forest import ForestInverseMass
from newton._src.solvers.phoenx.tests import test_contact_coupling as contact_coupling


class TestReducedForest(unittest.TestCase):
    def test_forest_contact_matches_scalar_fallback(self):
        """Match the forest response to the scalar contact fallback."""
        reference = contact_coupling.TestContactCoupling()
        reference.test_cross_articulation_friction_does_not_add_energy()
        forest = contact_coupling.TestContactCoupling()
        forest._use_forest = True
        forest.test_cross_articulation_friction_does_not_add_energy()
        # The scalar polarization and direct matrix products have different
        # float32 accumulation orders under 1000:1 inertia anisotropy.
        np.testing.assert_allclose(forest._last_velocity, reference._last_velocity, rtol=2.0e-5, atol=2.0e-6)

    def test_forest_cross_articulation_friction_does_not_add_energy(self):
        """Preserve dissipative friction across separate articulations."""
        case = contact_coupling.TestContactCoupling()
        case._use_forest = True
        case.test_cross_articulation_friction_does_not_add_energy()

    def test_forest_self_contact_and_motor_conserve_momentum(self):
        """Conserve momentum during self-contact and motor actuation."""
        case = contact_coupling.TestContactCoupling()
        case._use_forest = True
        case.test_sustained_self_contact_and_motor_conserve_momentum()

    def test_batched_inverse_mass_matches_independent_tree_solves(self):
        """Match batched inverse mass to independent tree responses."""
        if not wp.is_cuda_available():
            self.skipTest("CUDA required")
        builder = newton.ModelBuilder(gravity=wp.vec3(0.0))
        root = builder.add_link(mass=2.0, inertia=wp.mat33(0.03, 0.0, 0.0, 0.0, 0.02, 0.0, 0.0, 0.0, 0.04))
        child = builder.add_link(mass=1.0, inertia=wp.mat33(0.01, 0.0, 0.0, 0.0, 0.03, 0.0, 0.0, 0.0, 0.03))
        joints = [builder.add_joint_free(root)]
        joints.append(
            builder.add_joint_revolute(
                root, child, axis=newton.Axis.Y, parent_xform=wp.transform(wp.vec3(0.2, 0.1, 0.3), wp.quat_identity())
            )
        )
        builder.add_articulation(joints)
        independent = builder.add_link(mass=0.7, inertia=wp.mat33(0.04, 0.0, 0.0, 0.0, 0.02, 0.0, 0.0, 0.0, 0.05))
        builder.add_articulation([builder.add_joint_free(independent)])
        model = builder.finalize(device="cuda:0")
        state = model.state()
        newton.eval_fk(model, state.joint_q, state.joint_qd, state)
        system = ReducedArticulationSystem(model)
        system.factor(state)
        forest = ForestInverseMass(system)
        forest.refresh()
        matrix = forest.matrix.numpy()[: forest.dof_count, : forest.dof_count]
        indices = forest.compact_dofs.numpy()
        for column, dof in enumerate(indices):
            force = np.zeros(model.joint_dof_count, dtype=np.float32)
            force[dof] = 1.0
            response = system.solve_generalized(wp.array(force, device=model.device)).numpy()
            np.testing.assert_allclose(matrix[:, column], response[indices], rtol=2.0e-6, atol=2.0e-6)
        np.testing.assert_allclose(matrix, matrix.T, rtol=2.0e-6, atol=2.0e-6)
        np.testing.assert_array_equal(matrix[:7, 7:], np.zeros((7, 6)))
        self.assertGreater(float(np.linalg.eigvalsh(matrix).min()), 0.0)


if __name__ == "__main__":
    unittest.main()
