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

from newton._src.solvers.phoenx.benchmarks.scenarios import g1_flat, h1_flat

SCENARIOS = {
    "g1_flat": g1_flat,
    "h1_flat": h1_flat,
}

__all__ = ["SCENARIOS", "g1_flat", "h1_flat"]
