# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Convenience entry point for running every jitter unit test.

Discovers and runs every ``test_*.py`` module in this directory (except
this file itself) in a single ``unittest`` invocation. Intended for
local development -- ``uv run --extra dev -m unittest
newton._src.solvers.phoenx.tests.test_run_all`` is much faster than
``newton.tests`` because it skips the rest of the project's test
suite (and the associated solver/example warm-up).

Run as::

    uv run --extra dev -m unittest newton._src.solvers.phoenx.tests.test_run_all

or directly::

    uv run --extra dev python -m newton._src.solvers.phoenx.tests.test_run_all
"""

from __future__ import annotations

import os
import unittest


def load_tests(loader: unittest.TestLoader, standard_tests, pattern):
    """unittest protocol hook: discover sibling ``test_*.py`` modules.

    Picked up automatically by ``unittest`` when this module is given as
    the test target (``unittest`` calls ``load_tests`` if defined).
    Enumerates ``test_*.py`` files in this directory and loads each as
    ``newton._src.solvers.phoenx.<stem>``; this avoids
    ``loader.discover`` (which insists on its start dir being a
    top-level importable package) and keeps every test reachable via
    its real dotted module name.
    """
    here = os.path.dirname(os.path.abspath(__file__))
    package = __name__.rsplit(".", 1)[0]
    self_stem = os.path.splitext(os.path.basename(__file__))[0]

    suite = unittest.TestSuite()
    for fname in sorted(os.listdir(here)):
        if not fname.startswith("test_") or not fname.endswith(".py"):
            continue
        stem = fname[:-3]
        if stem == self_stem:
            continue
        suite.addTests(loader.loadTestsFromName(f"{package}.{stem}"))
    return suite


if __name__ == "__main__":
    unittest.main(verbosity=2)
