# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Convenience entry point for running every PhoenX unit test.

Discovers and runs every ``test_*.py`` module in this directory (except
this file itself) in a single ``unittest`` invocation. Intended for
local development -- ``uv run --extra dev -m unittest
newton._src.solvers.phoenx.tests.test_run_all`` is much faster than
``newton.tests`` because it skips the rest of the project's test
suite (and the associated solver/example warm-up).

Run as::

    uv run --extra dev -m unittest newton._src.solvers.phoenx.tests.test_run_all

or directly (preferred -- this branch enables the timing report)::

    uv run --extra dev python -m newton._src.solvers.phoenx.tests.test_run_all

The PhoenX solver is GPU-only by design (the whole hot path is built on
CUDA graph capture and warp kernel launches, with no functional CPU
fallback for the big kernels). To make CI / dev failures obvious early,
this entry point hard-fails when no CUDA device is available *before*
loading any test modules.

When invoked as a script (``python -m`` form), it also writes a per-test
timing report to ``test_run_all_report.txt`` next to the working
directory (override via ``NEWTON_PHOENX_TIMING_REPORT``) so regressions
in test wall-time -- typically a graph-capture fallback to eager
stepping -- are caught at a glance. Sample report row::

    1.234s  ok       newton._src.solvers.phoenx.tests.test_beam_joint.TestBeamAnalytical.test_undamped_period_within_5pct
"""

from __future__ import annotations

import os
import sys
import time
import unittest

import warp as wp


_REPORT_PATH_ENV = "NEWTON_PHOENX_TIMING_REPORT"
_DEFAULT_REPORT_FILENAME = "test_run_all_report.txt"


def _require_cuda() -> None:
    """Hard-fail the whole suite if no CUDA device is present.

    PhoenX kernels do compile for CPU (Warp can target any device) but
    the integration tests deliberately exercise CUDA-graph capture
    paths and the per-test timing report assumes those are active. A
    silent CPU fallback would still pass a few cheap tests and then
    OOM / timeout deeper in the suite -- raising up front is friendlier.
    """
    try:
        device = wp.get_device()
    except Exception as exc:  # pragma: no cover -- warp init failure
        raise RuntimeError(
            "Could not query the active warp device. PhoenX tests require CUDA."
        ) from exc
    if not device.is_cuda:
        raise unittest.SkipTest(
            f"PhoenX tests require a CUDA device (active device: {device.name!r}). "
            "Install a CUDA-capable Warp build (e.g. ``uv sync --extra dev`` on a "
            "machine with an NVIDIA GPU) and re-run."
        )


def load_tests(loader: unittest.TestLoader, standard_tests, pattern):
    """unittest protocol hook: discover sibling ``test_*.py`` modules.

    Picked up automatically by ``unittest`` when this module is given
    as the test target (``unittest`` calls ``load_tests`` if defined).
    Enumerates ``test_*.py`` files in this directory and loads each as
    ``newton._src.solvers.phoenx.<stem>``; this avoids
    ``loader.discover`` (which insists on its start dir being a
    top-level importable package) and keeps every test reachable via
    its real dotted module name.

    Hard-fails up front if no CUDA device is available -- see
    :func:`_require_cuda`.
    """
    _require_cuda()

    here = os.path.dirname(os.path.abspath(__file__))
    # Hard-coded so the protocol hook works regardless of how this
    # module was loaded (``python -m`` -> ``__name__ == "__main__"``;
    # ``-m unittest <dotted>`` -> ``__name__`` is the dotted path).
    package = "newton._src.solvers.phoenx.tests"
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


class _TimingTestResult(unittest.TextTestResult):
    """``TextTestResult`` subclass that records per-test wall-time.

    We can't reuse stock ``TextTestResult`` because it doesn't expose
    a clean before/after hook around the ``test`` callable -- only the
    ``startTest`` / ``stopTest`` pair, which is exactly what we need.
    Recording at this level (rather than wrapping each ``test_*``
    method) means the time covers the whole frame including
    setup / teardown and any module-level import work attributed to
    the *first* test loaded from a given module.

    Records a short outcome tag (``ok`` / ``fail`` / ``error`` /
    ``skip`` / ``xfail`` / ``xok``) so the report is greppable for
    regressions without re-running.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # List of (test_id, elapsed_seconds, outcome) tuples; preserved
        # in start order. ``unittest`` runs single-threaded by default
        # in ``unittest.main``, so no locking required.
        self.timings: list[tuple[str, float, str]] = []
        self._t_start: float | None = None
        self._current_outcome: str = "ok"

    def startTest(self, test: unittest.TestCase) -> None:
        super().startTest(test)
        self._t_start = time.perf_counter()
        self._current_outcome = "ok"

    def stopTest(self, test: unittest.TestCase) -> None:
        elapsed = (time.perf_counter() - self._t_start) if self._t_start is not None else 0.0
        self.timings.append((test.id(), elapsed, self._current_outcome))
        self._t_start = None
        super().stopTest(test)

    def addError(self, test: unittest.TestCase, err) -> None:
        self._current_outcome = "error"
        super().addError(test, err)

    def addFailure(self, test: unittest.TestCase, err) -> None:
        self._current_outcome = "fail"
        super().addFailure(test, err)

    def addSkip(self, test: unittest.TestCase, reason: str) -> None:
        self._current_outcome = "skip"
        super().addSkip(test, reason)

    def addExpectedFailure(self, test: unittest.TestCase, err) -> None:
        self._current_outcome = "xfail"
        super().addExpectedFailure(test, err)

    def addUnexpectedSuccess(self, test: unittest.TestCase) -> None:
        self._current_outcome = "xok"
        super().addUnexpectedSuccess(test)


class _TimingTestRunner(unittest.TextTestRunner):
    """``TextTestRunner`` that pairs :class:`_TimingTestResult` with a
    final report dump. Writes to ``report_path`` after the run
    finishes (regardless of pass / fail) so the report is always
    available for triage.
    """

    resultclass = _TimingTestResult

    def __init__(self, *args, report_path: str | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        self._report_path = report_path

    def run(self, test):
        result = super().run(test)
        if self._report_path is not None and isinstance(result, _TimingTestResult):
            try:
                _write_timing_report(self._report_path, result.timings)
            except OSError as exc:  # pragma: no cover -- non-fatal
                print(
                    f"\nWARNING: could not write timing report to {self._report_path!r}: {exc}",
                    file=sys.stderr,
                )
            else:
                print(f"\nTiming report: {self._report_path}")
        return result


def _write_timing_report(path: str, timings: list[tuple[str, float, str]]) -> None:
    """Dump ``timings`` to ``path`` in two sections: chronological
    (start order) and slowest-first (top 30). Both sections share
    columns ``elapsed | outcome | test_id`` so they're easy to grep
    and diff between runs.
    """
    total = sum(t for _, t, _ in timings)
    sorted_by_time = sorted(timings, key=lambda row: row[1], reverse=True)

    with open(path, "w", encoding="utf-8") as f:
        f.write("# PhoenX unit test timing report\n")
        f.write(f"# Tests recorded: {len(timings)}\n")
        f.write(f"# Total wall-time: {total:.3f}s\n")
        f.write("# Columns: elapsed_s outcome test_id\n")
        f.write("#\n")
        f.write("# --- Slowest 30 tests ---\n")
        for tid, elapsed, outcome in sorted_by_time[:30]:
            f.write(f"{elapsed:8.3f}  {outcome:6s}  {tid}\n")
        f.write("#\n")
        f.write("# --- All tests in run order ---\n")
        for tid, elapsed, outcome in timings:
            f.write(f"{elapsed:8.3f}  {outcome:6s}  {tid}\n")


def _resolve_report_path() -> str:
    """Honor ``$NEWTON_PHOENX_TIMING_REPORT`` so CI can route the
    report to a tracked artifact dir without code changes; default to
    a relative file in the current working directory.
    """
    return os.environ.get(_REPORT_PATH_ENV) or os.path.abspath(_DEFAULT_REPORT_FILENAME)


def main() -> None:
    """Run all sibling ``test_*.py`` modules with :class:`_TimingTestRunner`.

    Invoked when this module is run as a script (``python -m
    newton._src.solvers.phoenx.tests.test_run_all``). The unittest
    ``-m unittest ... test_run_all`` form goes through stock
    ``unittest.main`` instead and skips the timing report -- intended:
    that path is for IDE / pre-commit integration where we don't want
    to clobber the working dir with a report file.
    """
    _require_cuda()

    here = os.path.dirname(os.path.abspath(__file__))
    # When invoked via ``python -m``, ``__name__`` is ``"__main__"`` so
    # we cannot derive the package from it -- hard-code the dotted path
    # to this directory's package instead. Keeps behaviour identical
    # whether the module was loaded as a script or as
    # ``newton._src.solvers.phoenx.tests.test_run_all``.
    package = "newton._src.solvers.phoenx.tests"
    self_stem = os.path.splitext(os.path.basename(__file__))[0]

    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    for fname in sorted(os.listdir(here)):
        if not fname.startswith("test_") or not fname.endswith(".py"):
            continue
        stem = fname[:-3]
        if stem == self_stem:
            continue
        suite.addTests(loader.loadTestsFromName(f"{package}.{stem}"))

    runner = _TimingTestRunner(verbosity=2, report_path=_resolve_report_path())
    result = runner.run(suite)
    sys.exit(0 if result.wasSuccessful() else 1)


if __name__ == "__main__":
    main()
