# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""G1-flat scene from the FeatherPGS nightly benchmark."""

from newton._src.solvers.phoenx.benchmarks.runner import SceneHandle
from newton._src.solvers.phoenx.benchmarks.scenarios import feather_pgs_common


def build(
    num_worlds: int,
    solver_name: str,
    substeps: int,
    solver_iterations: int,
    *,
    articulation_mode: str = "maximal",
) -> SceneHandle:
    """Build the copied FeatherPGS G1-flat benchmark."""
    return feather_pgs_common.build(
        "feather_pgs_g1_flat",
        num_worlds,
        solver_name,
        substeps,
        solver_iterations,
        articulation_mode=articulation_mode,
    )
