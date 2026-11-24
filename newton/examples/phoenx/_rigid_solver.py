# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Share the large-world colored joint/contact solver configuration."""

import newton.solvers


def make_colored_solver(
    model,
    pipeline,
    device,
    *,
    substeps=4,
    solver_iterations=10,
    direct_joint_mask=None,
    direct_joint_projection_passes=None,
    contact_chunk_size=0,
):
    """Use physical bodies in regular colors and mass copies in overflow."""
    return newton.solvers.SolverPhoenX(
        model,
        collision_pipeline=pipeline,
        substeps=substeps,
        solver_iterations=solver_iterations,
        velocity_iterations=1,
        step_layout="single_world",
        mass_splitting=True,
        mass_splitting_overflow_only=True,
        max_colored_partitions=9,
        joint_mode="maximal_pgs",
        sleeping_velocity_threshold=0.0,
        direct_joint_mask=direct_joint_mask,
        direct_joint_projection_passes=direct_joint_projection_passes,
        contact_chunk_size=contact_chunk_size,
    )
