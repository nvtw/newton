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

import newton
import newton.examples
from newton.examples.cable.example_cable_pile import Example as ExampleVBD


class Example(ExampleVBD):
    """Exact VBD cable-pile scene translated to PhoenX."""

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
        """Replace only the VBD solver with substepped PhoenX PGS."""
        return newton.solvers.SolverPhoenX(
            self.model,
            substeps=1,
            solver_iterations=self.sim_iterations,
            velocity_iterations=1,
            prepare_refresh_stride=1,
        )


if __name__ == "__main__":
    viewer, args = newton.examples.init()
    newton.examples.run(Example(viewer, args), args)
