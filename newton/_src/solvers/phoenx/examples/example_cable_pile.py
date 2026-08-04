# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example PhoenX Cable Pile
#
# Runs the canonical VBD cable-pile scene with SolverPhoenX. Scene
# construction, physical properties, contact matching, simulation cadence,
# rendering, and validation are inherited unchanged from the VBD example.
#
# Run:
#   python -m newton._src.solvers.phoenx.examples.example_cable_pile
###########################################################################

import numpy as np

import newton
import newton.examples
from newton.examples.cable.example_cable_pile import Example as ExampleVBD


class Example(ExampleVBD):
    """Exact VBD cable-pile scene translated to PhoenX."""

    def __init__(self, *args, **kwargs):
        slope_enabled = kwargs.get("slope_enabled", args[2] if len(args) > 2 else False)
        self._check_lateral_drift = not slope_enabled
        super().__init__(*args, **kwargs)

    def finalize_model(self, builder):
        """Skip the explicit all-pairs list consumed only by VBD."""
        return builder.finalize(skip_shape_contact_pairs=True)

    def create_collision_pipeline(self):
        """Use bounded SAP broad phase for the unchanged 4,000-body scene."""
        return newton.CollisionPipeline(
            self.model,
            contact_matching="latest",
            broad_phase="sap",
            shape_pairs_max=500_000,
        )

    def create_solver(self):
        """Use direct PhoenX cable equalities with PGS contacts."""
        return newton.solvers.SolverPhoenX(
            self.model,
            # The inherited frame loop supplies ten contact-refresh substeps.
            substeps=1,
            solver_iterations=self.sim_iterations,
            velocity_iterations=1,
            prepare_refresh_stride=1,
        )

    def test_final(self):
        """Verify stability and reject systematic lateral pile creep."""
        super().test_final()
        if self._check_lateral_drift:
            positions = self.state_0.body_q.numpy()[:, :3]
            mass = self.model.body_mass.numpy()
            center_xy = np.sum(positions[:, :2] * mass[:, None], axis=0) / np.sum(mass)
            assert np.linalg.norm(center_xy) < 1.0e-3, f"Cable pile drifted sideways: center_xy={center_xy}"


if __name__ == "__main__":
    viewer, args = newton.examples.init()
    newton.examples.run(Example(viewer, args), args)
