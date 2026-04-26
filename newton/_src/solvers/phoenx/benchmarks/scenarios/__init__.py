# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Scene factories for the PhoenX benchmark harness.

Each module exposes a :func:`build` callable returning a
:class:`~newton._src.solvers.phoenx.benchmarks.runner.SceneHandle`.
Scene builders are intentionally thin clones of the corresponding
``newton.examples.robot_*`` examples (headless, no viewer, no test
tracking) so solver parameters (``substeps``, ``solver_iterations``)
can be varied for sweeps.
"""

from newton._src.solvers.phoenx.benchmarks.scenarios import (
    big_box_grid,
    g1_flat,
    h1_flat,
    tower_grid,
)

SCENARIOS = {
    "big_box_grid": big_box_grid,
    "g1_flat": g1_flat,
    "h1_flat": h1_flat,
    "tower_grid": tower_grid,
}

__all__ = ["SCENARIOS", "big_box_grid", "g1_flat", "h1_flat", "tower_grid"]
