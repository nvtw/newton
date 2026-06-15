# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Opt-in gate for PhoenX tests that run the MuJoCo solver alongside
PhoenX (parity / cross-solver-regression suites).

These tests instantiate :class:`SolverMuJoCo`, which pulls in
``mujoco_warp`` and JIT-compiles its CUDA kernels on first use --
notably the Cholesky factorise / solve / gradient kernels, each of
which costs 10-12 seconds per cold-cache test run. A full MuJoCo
parity sweep adds 30-60 s of compile time and another minute of
solver-vs-solver stepping, dwarfing the rest of the unit test
suite.

Skipping by default keeps the everyday ``test_run_all`` invocation
under a minute on a hot cache; opt in only when validating a
cross-solver behaviour change. Set the env var to ``"1"`` to enable.

Usage::

    from newton._src.solvers.phoenx.tests._mujoco_skip import skip_unless_mujoco_opt_in


    @skip_unless_mujoco_opt_in
    class TestSomeMuJoCoParity(unittest.TestCase): ...

or, for a whole module, add at module scope::

    setUpModule = require_mujoco_opt_in
"""

from __future__ import annotations

import os
import unittest

__all__ = [
    "MUJOCO_OPT_IN_ENV",
    "require_mujoco_opt_in",
    "skip_unless_mujoco_opt_in",
]

#: Env var that opts a test process into running MuJoCo-dependent tests.
#: Set to ``"1"`` (or any truthy non-empty value other than ``"0"``).
MUJOCO_OPT_IN_ENV = "NEWTON_PHOENX_RUN_MUJOCO_TESTS"

_SKIP_REASON = (
    "MuJoCo cross-solver test skipped by default. These tests instantiate "
    "SolverMuJoCo for parity comparison, which pulls in mujoco_warp and "
    "JIT-compiles its CUDA kernels (Cholesky factorise / solve, gradient "
    "updates) at 10-12 s per kernel on a cold cache -- a full MuJoCo "
    "parity sweep adds 30-60 s of compile + minutes of stepping. To run "
    f"them set {MUJOCO_OPT_IN_ENV}=1 in the environment. "
    "Discouraged for routine unit-test runs; only flip on when validating "
    "a cross-solver behaviour change and you can spare several minutes "
    "for the cold-cache compile."
)


def _opt_in_active() -> bool:
    val = os.environ.get(MUJOCO_OPT_IN_ENV, "")
    return val not in ("", "0", "false", "False", "no", "No")


def require_mujoco_opt_in() -> None:
    """``setUpModule`` hook: skip every test in the module unless the
    user opted in via :data:`MUJOCO_OPT_IN_ENV`.
    """
    if not _opt_in_active():
        raise unittest.SkipTest(_SKIP_REASON)


def skip_unless_mujoco_opt_in(cls):
    """Class decorator: skip the wrapped :class:`TestCase` (and all of
    its tests) unless :data:`MUJOCO_OPT_IN_ENV` is set."""
    return unittest.skipUnless(_opt_in_active(), _SKIP_REASON)(cls)
