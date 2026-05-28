# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Convenience entry point for running every PhoenX unit test.

Discovers and runs every ``test_*.py`` module in this directory plus the
nested mass-splitting unit tests in a single ``unittest`` invocation.
Intended for local development -- ``uv run --extra dev -m unittest
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
this entry point hard-fails when no CUDA graph-capture-capable device is
available *before* loading any test modules.

When invoked as a script (``python -m`` form), it also writes a per-test
timing report to ``test_run_all_report.txt`` next to the working
directory (override via ``NEWTON_PHOENX_TIMING_REPORT``) so regressions
in test wall-time -- typically a graph-capture fallback to eager
stepping -- are caught at a glance. Sample report row::

    1.234s  ok       newton._src.solvers.phoenx.tests.test_cable_joint.TestCableAnalytical.test_undamped_period_within_5pct
"""

from __future__ import annotations

import os
import sys
import time
import unittest

from newton._src.solvers.phoenx.tests._test_helpers import require_cuda_graph_capture

_REPORT_PATH_ENV = "NEWTON_PHOENX_TIMING_REPORT"
_DEFAULT_REPORT_FILENAME = "test_run_all_report.txt"

#: Defensive guard: any future test module added here is skipped by
#: ``test_run_all``. Use it to exclude tests that instantiate phoenx
#: ``Example`` classes -- those should never run as part of the unit
#: test sweep; examples validate themselves via their ``test_final`` /
#: ``test_post_step`` hooks when actually run.
_EXAMPLE_RUNNING_TEST_MODULES: frozenset[str] = frozenset()


def _require_cuda_graph_capture() -> None:
    require_cuda_graph_capture()


def _iter_test_module_names(self_stem: str) -> list[str]:
    """Return dotted PhoenX test module names covered by this runner."""
    here = os.path.dirname(os.path.abspath(__file__))
    roots = (
        (here, "newton._src.solvers.phoenx.tests", {self_stem}),
        (
            os.path.abspath(os.path.join(here, os.pardir, "mass_splitting", "tests")),
            "newton._src.solvers.phoenx.mass_splitting.tests",
            frozenset[str](),
        ),
    )

    modules: list[str] = []
    for root, package, skip_stems in roots:
        for fname in sorted(os.listdir(root)):
            if not fname.startswith("test_") or not fname.endswith(".py"):
                continue
            stem = fname[:-3]
            if stem in skip_stems or stem in _EXAMPLE_RUNNING_TEST_MODULES:
                continue
            modules.append(f"{package}.{stem}")
    return modules


def load_tests(loader: unittest.TestLoader, standard_tests, pattern):
    """unittest protocol hook: discover sibling ``test_*.py`` modules.

    Picked up automatically by ``unittest`` when this module is given
    as the test target (``unittest`` calls ``load_tests`` if defined).
    Enumerates the configured PhoenX test roots explicitly instead of
    using ``loader.discover`` (which insists on its start dir being a
    top-level importable package) and keeps every test reachable via
    its real dotted module name.

    Hard-fails up front if CUDA graph capture is unavailable -- see
    :func:`_require_cuda_graph_capture`.
    """
    _require_cuda_graph_capture()

    self_stem = os.path.splitext(os.path.basename(__file__))[0]

    suite = unittest.TestSuite()
    for module_name in _iter_test_module_names(self_stem):
        suite.addTests(loader.loadTestsFromName(module_name))
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
        # Optional handle for live per-test flushing. Set by the runner
        # before the suite runs; if non-None we append one row per test
        # in :meth:`stopTest` and ``fsync`` so a Ctrl-C / OOM mid-suite
        # still leaves a parseable timing log on disk. The runner
        # rewrites the file with the structured (slowest-30 + chrono)
        # report at the end of a clean run.
        self._live_handle = None

    def startTest(self, test: unittest.TestCase) -> None:
        super().startTest(test)
        self._t_start = time.perf_counter()
        self._current_outcome = "ok"

    def stopTest(self, test: unittest.TestCase) -> None:
        elapsed = (time.perf_counter() - self._t_start) if self._t_start is not None else 0.0
        self.timings.append((test.id(), elapsed, self._current_outcome))
        self._t_start = None
        if self._live_handle is not None:
            try:
                self._live_handle.write(f"{elapsed:8.3f}  {self._current_outcome:6s}  {test.id()}\n")
                self._live_handle.flush()
                os.fsync(self._live_handle.fileno())
            except OSError:  # pragma: no cover -- non-fatal
                pass
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

    def _makeResult(self):
        # Stock ``TextTestRunner.run`` calls ``self._makeResult()`` and
        # uses the returned object as the run's result sink. We hook in
        # here so we can attach the live-flush handle to *the same*
        # result instance ``run`` will then drive -- doing it after
        # ``super().run()`` would be too late (every ``stopTest`` call
        # has already fired).
        result = super()._makeResult()
        if self._report_path is not None and isinstance(result, _TimingTestResult):
            try:
                # Truncate any previous run's report so partial results
                # never get mixed in. Header lets a hung-then-killed run
                # still parse via ``grep -v '^#'``.
                handle = open(self._report_path, "w", encoding="utf-8")
                handle.write("# PhoenX unit test timing report (live)\n")
                handle.write("# Columns: elapsed_s outcome test_id\n")
                handle.write("#\n")
                handle.flush()
                result._live_handle = handle
            except OSError as exc:  # pragma: no cover -- non-fatal
                print(
                    f"\nWARNING: could not open live timing report at {self._report_path!r}: {exc}",
                    file=sys.stderr,
                )
        return result

    def run(self, test):
        result = super().run(test)
        if isinstance(result, _TimingTestResult) and result._live_handle is not None:
            try:
                result._live_handle.close()
            except OSError:  # pragma: no cover -- non-fatal
                pass
            result._live_handle = None
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
    """Run all PhoenX ``test_*.py`` modules with :class:`_TimingTestRunner`.

    Invoked when this module is run as a script (``python -m
    newton._src.solvers.phoenx.tests.test_run_all``). The unittest
    ``-m unittest ... test_run_all`` form goes through stock
    ``unittest.main`` instead and skips the timing report -- intended:
    that path is for IDE / pre-commit integration where we don't want
    to clobber the working dir with a report file.
    """
    _require_cuda_graph_capture()

    self_stem = os.path.splitext(os.path.basename(__file__))[0]

    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    for module_name in _iter_test_module_names(self_stem):
        suite.addTests(loader.loadTestsFromName(module_name))

    runner = _TimingTestRunner(verbosity=2, report_path=_resolve_report_path())
    result = runner.run(suite)
    sys.exit(0 if result.wasSuccessful() else 1)


if __name__ == "__main__":
    main()
