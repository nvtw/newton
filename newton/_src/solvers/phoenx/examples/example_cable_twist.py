# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example PhoenX Cable Twist
#
# Runs the canonical VBD cable-twist scene with SolverPhoenX. Scene
# construction, physical properties, kinematic drive, simulation cadence,
# rendering, and validation are inherited unchanged from the VBD example.
#
# Run:
#   python -m newton._src.solvers.phoenx.examples.example_cable_twist
###########################################################################

import newton
import newton.examples
from newton.examples.cable.example_cable_twist import Example as ExampleVBD


class Example(ExampleVBD):
    """Exact VBD cable-twist scene translated to PhoenX."""

    def create_solver(self):
        """Replace only the VBD solver with substepped PhoenX PGS."""
        return newton.solvers.SolverPhoenX(
            self.model,
            substeps=1,
            solver_iterations=self.sim_iterations,
            velocity_iterations=1,
            prepare_refresh_stride=1,
        )

    def create_collision_pipeline(self):
        """Enable the contact matching required by PhoenX."""
        return newton.CollisionPipeline(self.model, contact_matching="latest")

    def update_solver_contact_history(self, refresh_contacts):
        """PhoenX ingests the current contact buffer on every solver call."""


if __name__ == "__main__":
    viewer, args = newton.examples.init()
    newton.examples.run(Example(viewer, args), args)
