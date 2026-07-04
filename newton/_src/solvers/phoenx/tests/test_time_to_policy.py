# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import math
import tempfile
import unittest
from pathlib import Path

from newton._src.solvers.phoenx.benchmarks.bench_g1_train_to_gate import _updated_pass_streak
from newton._src.solvers.phoenx.benchmarks.bench_time_to_policy import (
    TrialOutcome,
    _child_command,
    _validate_forwarded_args,
    _write_json,
    summarize_trials,
)


def _trial(
    index: int,
    seconds: float,
    *,
    passed: bool,
    censored: bool | None = None,
) -> TrialOutcome:
    return TrialOutcome(
        task="g1",
        trial_index=index,
        train_seed=10 + index,
        gate_seed=100 + index,
        process_wall_seconds=seconds,
        charged_seconds=seconds if passed else 100.0,
        pass_gate=passed,
        censored=not passed if censored is None else censored,
        timed_out=False,
        return_code=0,
        result_path=f"trial_{index}.json",
        log_path=f"trial_{index}.log",
    )


class TestTimeToPolicyStatistics(unittest.TestCase):
    def test_all_successes_use_mean_success_time(self):
        summary = summarize_trials(
            [_trial(0, 20.0, passed=True), _trial(1, 30.0, passed=True)], 100.0, bootstrap_samples=0
        )

        self.assertEqual(summary["success_probability"], 1.0)
        self.assertEqual(summary["expected_attempts_to_success"], 1.0)
        self.assertEqual(summary["expected_seconds_to_success"], 25.0)
        self.assertEqual(summary["mean_charged_attempt_seconds"], 25.0)

    def test_misses_are_charged_as_complete_restarts(self):
        summary = summarize_trials(
            [
                _trial(0, 20.0, passed=True),
                _trial(1, 45.0, passed=False),
                _trial(2, 40.0, passed=True),
                _trial(3, 100.0, passed=False),
            ],
            100.0,
            bootstrap_samples=0,
        )

        self.assertEqual(summary["success_probability"], 0.5)
        self.assertEqual(summary["mean_success_seconds"], 30.0)
        self.assertEqual(summary["expected_seconds_to_success"], 130.0)
        self.assertEqual(summary["mean_charged_attempt_seconds"], 65.0)

    def test_zero_successes_report_infinite_expected_cost(self):
        summary = summarize_trials([_trial(0, 100.0, passed=False)], 100.0, bootstrap_samples=8)

        self.assertTrue(math.isinf(summary["expected_seconds_to_success"]))
        self.assertTrue(math.isinf(summary["expected_seconds_bootstrap95"][0]))
        self.assertTrue(math.isinf(summary["expected_seconds_bootstrap95"][1]))

    def test_infinite_estimate_is_written_as_strict_json_null(self):
        summary = summarize_trials([_trial(0, 100.0, passed=False)], 100.0, bootstrap_samples=0)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "summary.json"
            _write_json(path, summary)
            loaded = json.loads(path.read_text(encoding="utf-8"))

        self.assertTrue(loaded["expected_cost_is_infinite"])
        self.assertIsNone(loaded["expected_seconds_to_success"])

    def test_censored_nominal_pass_does_not_count(self):
        summary = summarize_trials(
            [_trial(0, 20.0, passed=True, censored=True), _trial(1, 30.0, passed=True)],
            100.0,
            bootstrap_samples=0,
        )

        self.assertEqual(summary["success_count"], 1)
        self.assertEqual(summary["expected_seconds_to_success"], 130.0)

    def test_bootstrap_is_reproducible(self):
        trials = [_trial(0, 20.0, passed=True), _trial(1, 100.0, passed=False), _trial(2, 30.0, passed=True)]
        first = summarize_trials(trials, 100.0, bootstrap_samples=50, bootstrap_seed=7)
        second = summarize_trials(trials, 100.0, bootstrap_samples=50, bootstrap_seed=7)

        self.assertEqual(first["expected_seconds_bootstrap95"], second["expected_seconds_bootstrap95"])


class TestTimeToPolicyProtocol(unittest.TestCase):
    def test_pass_streak_resets_on_failure(self):
        streak = 0
        for passed in (True, True, False, True):
            streak = _updated_pass_streak(passed, streak)

        self.assertEqual(streak, 1)

    def test_runner_owns_seed_and_output_arguments(self):
        for option in ("--seed", "--seed=12", "--json-output", "--required-consecutive-passes=3"):
            with self.subTest(option=option), self.assertRaises(ValueError):
                _validate_forwarded_args([option])

    def test_child_command_contains_frozen_trial_protocol(self):
        command = _child_command(
            task="g1",
            train_seed=11,
            gate_seed=1000,
            required_consecutive_passes=2,
            result_path=Path("result.json"),
            checkpoint_path=Path("checkpoint_{iteration}.npz"),
            forwarded_args=("--world-count", "64"),
        )

        self.assertIn("--fail-on-miss", command)
        self.assertEqual(command[command.index("--seed") + 1], "11")
        self.assertEqual(command[command.index("--gate-seed") + 1], "1000")
        self.assertEqual(command[command.index("--required-consecutive-passes") + 1], "2")
        self.assertEqual(command[-2:], ["--world-count", "64"])


if __name__ == "__main__":
    unittest.main()
