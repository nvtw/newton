# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import math
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

from newton._src.solvers.phoenx.benchmarks.bench_dr_legs_hold_train_to_gate import (
    StatsEvaluateDrLegsHold,
)
from newton._src.solvers.phoenx.benchmarks.bench_dr_legs_hold_train_to_gate import (
    _make_parser as make_dr_legs_parser,
)
from newton._src.solvers.phoenx.benchmarks.bench_dr_legs_hold_train_to_gate import (
    check_gate as check_dr_legs_gate,
)
from newton._src.solvers.phoenx.benchmarks.bench_g1_train_to_gate import (
    _screen_promising,
    _screen_rejects_early,
    _updated_pass_streak,
)
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
    failure_charge: float = 100.0,
    early_stopped: bool = False,
) -> TrialOutcome:
    is_censored = not passed if censored is None else censored
    return TrialOutcome(
        task="g1",
        trial_index=index,
        train_seed=10 + index,
        gate_seed=100 + index,
        process_wall_seconds=seconds,
        charged_seconds=seconds if passed and not is_censored else failure_charge,
        pass_gate=passed,
        censored=is_censored,
        timed_out=False,
        return_code=0,
        result_path=f"trial_{index}.json",
        log_path=f"trial_{index}.log",
        early_stopped=early_stopped,
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

    def test_early_restarts_use_observed_failure_cost(self):
        summary = summarize_trials(
            [
                _trial(0, 20.0, passed=True),
                _trial(1, 40.0, passed=False, failure_charge=40.0, early_stopped=True),
            ],
            100.0,
            bootstrap_samples=0,
        )

        self.assertEqual(summary["early_stopped_count"], 1)
        self.assertEqual(summary["mean_failure_seconds"], 40.0)
        self.assertEqual(summary["expected_seconds_to_success"], 60.0)

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

    def test_cheap_g1_screen_promotes_only_promising_checkpoints(self):
        args = SimpleNamespace(
            screen_trigger_battery_perf=0.85,
            min_battery_perf=0.90,
            max_battery_falls=1,
        )

        self.assertFalse(_screen_promising(SimpleNamespace(battery_perf=0.84, battery_falls=0), args))
        self.assertTrue(_screen_promising(SimpleNamespace(battery_perf=0.86, battery_falls=1), args))
        self.assertFalse(_screen_promising(SimpleNamespace(battery_perf=0.90, battery_falls=2), args))
        args.min_battery_perf = -1.0
        self.assertTrue(_screen_promising(SimpleNamespace(battery_perf=0.0, battery_falls=0), args))

    def test_early_g1_screen_rejects_only_frozen_failures(self):
        args = SimpleNamespace(early_reject_min_battery_perf=0.80, max_battery_falls=1)

        self.assertTrue(_screen_rejects_early(SimpleNamespace(battery_perf=0.79, battery_falls=0), args))
        self.assertFalse(_screen_rejects_early(SimpleNamespace(battery_perf=0.81, battery_falls=1), args))
        self.assertTrue(_screen_rejects_early(SimpleNamespace(battery_perf=0.90, battery_falls=2), args))

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

    def test_anymal_command_uses_frozen_forward_gate(self):
        command = _child_command(
            task="anymal",
            train_seed=29,
            gate_seed=2000,
            required_consecutive_passes=2,
            result_path=Path("results/anymal.json"),
            checkpoint_path=Path("unused.npz"),
            forwarded_args=("--world-count", "128"),
        )

        self.assertIn("bench_anymal_train_to_gate", " ".join(command))
        self.assertEqual(command[command.index("--recipe") + 1], "forward")
        self.assertEqual(command[command.index("--execution-mode") + 1], "graph_leapfrog")
        self.assertEqual(command[command.index("--seed") + 1], "29")
        self.assertEqual(command[command.index("--gate-seed") + 1], "2000")
        self.assertEqual(command[-2:], ["--world-count", "128"])

    def test_dr_legs_command_uses_closed_loop_hold_gate(self):
        command = _child_command(
            task="dr_legs_hold",
            train_seed=47,
            gate_seed=3000,
            required_consecutive_passes=2,
            result_path=Path("results/dr_legs.json"),
            checkpoint_path=Path("checkpoint_{iteration}.npz"),
            forwarded_args=("--world-count", "64"),
        )

        self.assertIn("bench_dr_legs_hold_train_to_gate", " ".join(command))
        self.assertEqual(command[command.index("--seed") + 1], "47")
        self.assertEqual(command[command.index("--gate-seed") + 1], "3000")
        self.assertEqual(command[-2:], ["--world-count", "64"])

    def test_dr_legs_gate_includes_closed_loop_residual(self):
        args = make_dr_legs_parser().parse_args(["--checkpoint-path", "checkpoint.npz", "--json-output", "result.json"])
        stats = StatsEvaluateDrLegsHold(
            steps=250,
            fall_fraction=0.0,
            survival_fraction=1.0,
            mean_success=0.9,
            min_pelvis_height=0.24,
            max_pelvis_height=0.30,
            min_upright_cos=0.95,
            max_horizontal_drift=0.05,
            mean_action_rms=0.2,
            max_anchor_residual=5.0e-4,
            finite=True,
        )

        self.assertEqual(check_dr_legs_gate(stats, args), [])
        self.assertIn(
            "closed-loop anchor residual",
            check_dr_legs_gate(replace(stats, max_anchor_residual=2.0e-3), args),
        )


if __name__ == "__main__":
    unittest.main()
