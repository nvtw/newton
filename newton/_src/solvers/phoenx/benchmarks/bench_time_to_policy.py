# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Measure expected cold-process time to a qualified PhoenX policy.

Each trial starts the task benchmark in a fresh process, varies the training
seed, evaluates a frozen held-out seed bank, and charges misses, crashes, and
timeouts the complete restart cutoff. The primary estimate is the expected
wall time under that frozen restart policy, not successful-run throughput.

The task-specific arguments after ``--`` are forwarded to the child benchmark::

    uv run --extra dev -m newton._src.solvers.phoenx.benchmarks.bench_time_to_policy \
        --task g1 --train-seeds 11 29 47 --gate-seeds 1000 2000 \
        --restart-cutoff-seconds 180 -- \
        --world-count 8192 --chunk-iterations 25
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import random
import subprocess
import sys
import tempfile
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

_G1_MODULE = "newton._src.solvers.phoenx.benchmarks.bench_g1_train_to_gate"
_CONTROLLED_CHILD_OPTIONS = (
    "--checkpoint-path",
    "--fail-on-miss",
    "--gate-seed",
    "--json-output",
    "--keep-going-after-pass",
    "--required-consecutive-passes",
    "--seed",
)


@dataclass(frozen=True)
class TrialOutcome:
    """One independent cold-process training attempt."""

    task: str
    trial_index: int
    train_seed: int
    gate_seed: int
    process_wall_seconds: float
    charged_seconds: float
    pass_gate: bool
    censored: bool
    timed_out: bool
    return_code: int | None
    result_path: str
    log_path: str
    error: str | None = None


def _restart_expectation(trials: Sequence[TrialOutcome], cutoff_seconds: float) -> dict[str, Any]:
    """Estimate expected time until success under fixed-cutoff restarts."""
    if not trials:
        raise ValueError("at least one trial is required")
    if not math.isfinite(cutoff_seconds) or cutoff_seconds <= 0.0:
        raise ValueError("cutoff_seconds must be finite and positive")

    successful_times = [
        trial.process_wall_seconds
        for trial in trials
        if trial.pass_gate and not trial.censored and trial.process_wall_seconds <= cutoff_seconds
    ]
    trial_count = len(trials)
    success_count = len(successful_times)
    success_probability = success_count / trial_count
    mean_success_seconds = None if not successful_times else sum(successful_times) / success_count
    expected_seconds = (
        math.inf
        if mean_success_seconds is None
        else mean_success_seconds + cutoff_seconds * (1.0 - success_probability) / success_probability
    )
    mean_charged_seconds = (
        sum(
            trial.process_wall_seconds if is_successful else cutoff_seconds
            for trial, is_successful in (
                (item, item.pass_gate and not item.censored and item.process_wall_seconds <= cutoff_seconds)
                for item in trials
            )
        )
        / trial_count
    )

    # Wilson interval stays meaningful for the small adaptive-racing samples.
    z = 1.959963984540054
    denominator = 1.0 + z * z / trial_count
    center = (success_probability + z * z / (2.0 * trial_count)) / denominator
    radius = (
        z
        * math.sqrt(
            success_probability * (1.0 - success_probability) / trial_count + z * z / (4.0 * trial_count * trial_count)
        )
        / denominator
    )
    return {
        "trial_count": trial_count,
        "success_count": success_count,
        "censored_count": trial_count - success_count,
        "success_probability": success_probability,
        "success_probability_wilson95": [max(0.0, center - radius), min(1.0, center + radius)],
        "mean_success_seconds": mean_success_seconds,
        "mean_charged_attempt_seconds": mean_charged_seconds,
        "expected_attempts_to_success": math.inf if success_count == 0 else 1.0 / success_probability,
        "expected_seconds_to_success": expected_seconds,
        "expected_cost_is_infinite": math.isinf(expected_seconds),
        "restart_cutoff_seconds": cutoff_seconds,
    }


def _percentile(values: list[float], probability: float) -> float:
    values.sort()
    if len(values) == 1:
        return values[0]
    position = probability * (len(values) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return values[lower]
    fraction = position - lower
    if math.isinf(values[upper]):
        return math.inf
    return values[lower] * (1.0 - fraction) + values[upper] * fraction


def summarize_trials(
    trials: Sequence[TrialOutcome],
    cutoff_seconds: float,
    *,
    bootstrap_samples: int = 2000,
    bootstrap_seed: int = 20260704,
) -> dict[str, Any]:
    """Summarize trials and bootstrap uncertainty over complete attempts."""
    if bootstrap_samples < 0:
        raise ValueError("bootstrap_samples must be nonnegative")
    summary = _restart_expectation(trials, cutoff_seconds)
    if bootstrap_samples == 0:
        summary["expected_seconds_bootstrap95"] = None
        return summary

    rng = random.Random(bootstrap_seed)
    estimates: list[float] = []
    for _ in range(bootstrap_samples):
        sample = [trials[rng.randrange(len(trials))] for _ in trials]
        estimates.append(_restart_expectation(sample, cutoff_seconds)["expected_seconds_to_success"])
    summary["expected_seconds_bootstrap95"] = [
        _percentile(estimates.copy(), 0.025),
        _percentile(estimates.copy(), 0.975),
    ]
    summary["bootstrap_samples"] = bootstrap_samples
    summary["bootstrap_seed"] = bootstrap_seed
    return summary


def _json_compatible(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _json_compatible(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_compatible(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False
    ) as output:
        json.dump(_json_compatible(payload), output, indent=2, sort_keys=True, allow_nan=False)
        output.write("\n")
        temporary_path = Path(output.name)
    temporary_path.replace(path)


def _validate_forwarded_args(arguments: Sequence[str]) -> None:
    for argument in arguments:
        option = argument.split("=", 1)[0]
        if option in _CONTROLLED_CHILD_OPTIONS:
            raise ValueError(f"{option} is controlled by the time-to-policy runner")


def _child_command(
    *,
    task: str,
    train_seed: int,
    gate_seed: int,
    required_consecutive_passes: int,
    result_path: Path,
    checkpoint_path: Path,
    forwarded_args: Sequence[str],
) -> list[str]:
    if task != "g1":
        raise ValueError(f"unsupported task: {task}")
    return [
        sys.executable,
        "-m",
        _G1_MODULE,
        "--seed",
        str(train_seed),
        "--gate-seed",
        str(gate_seed),
        "--required-consecutive-passes",
        str(required_consecutive_passes),
        "--checkpoint-path",
        str(checkpoint_path),
        "--json-output",
        str(result_path),
        "--fail-on-miss",
        *forwarded_args,
    ]


def _run_trial(
    *,
    task: str,
    trial_index: int,
    train_seed: int,
    gate_seed: int,
    cutoff_seconds: float,
    required_consecutive_passes: int,
    run_directory: Path,
    forwarded_args: Sequence[str],
) -> TrialOutcome:
    stem = f"{task}_trial_{trial_index:03d}_train_{train_seed}_gate_{gate_seed}"
    result_path = run_directory / f"{stem}.json"
    log_path = run_directory / f"{stem}.log"
    checkpoint_path = run_directory / f"{stem}_checkpoint_{{iteration:06d}}.npz"
    command = _child_command(
        task=task,
        train_seed=train_seed,
        gate_seed=gate_seed,
        required_consecutive_passes=required_consecutive_passes,
        result_path=result_path,
        checkpoint_path=checkpoint_path,
        forwarded_args=forwarded_args,
    )

    timed_out = False
    return_code: int | None = None
    error: str | None = None
    started = time.perf_counter()
    try:
        with log_path.open("w", encoding="utf-8") as log:
            completed = subprocess.run(
                command,
                check=False,
                stdout=log,
                stderr=subprocess.STDOUT,
                timeout=cutoff_seconds,
            )
        return_code = completed.returncode
    except subprocess.TimeoutExpired:
        timed_out = True
        error = f"restart cutoff reached after {cutoff_seconds:.6g} seconds"
    except OSError as exception:
        error = f"failed to start child process: {exception}"
    process_seconds = min(time.perf_counter() - started, cutoff_seconds)

    child_result: dict[str, Any] = {}
    if result_path.is_file():
        try:
            child_result = json.loads(result_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exception:
            error = f"invalid child result: {exception}"
    elif error is None:
        error = "child process produced no result"

    pass_gate = bool(child_result.get("pass_gate", False)) and not timed_out and return_code == 0
    censored = not pass_gate
    if return_code not in (None, 0) and error is None:
        error = str(child_result.get("error", f"child exited with status {return_code}"))
    return TrialOutcome(
        task=task,
        trial_index=trial_index,
        train_seed=train_seed,
        gate_seed=gate_seed,
        process_wall_seconds=process_seconds,
        charged_seconds=process_seconds if pass_gate else cutoff_seconds,
        pass_gate=pass_gate,
        censored=censored,
        timed_out=timed_out,
        return_code=return_code,
        result_path=str(result_path),
        log_path=str(log_path),
        error=error,
    )


def _make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=("g1",), default="g1")
    parser.add_argument("--train-seeds", type=int, nargs="+", default=(11, 29, 47))
    parser.add_argument("--gate-seeds", type=int, nargs="+", default=(1000, 2000, 3000))
    parser.add_argument("--restart-cutoff-seconds", type=float, default=180.0)
    parser.add_argument("--required-consecutive-passes", type=int, default=2)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260704)
    parser.add_argument("--output", type=Path, default=Path("/tmp/phoenx_time_to_policy.json"))
    parser.add_argument("--run-directory", type=Path, default=Path("/tmp/phoenx_time_to_policy_trials"))
    parser.add_argument("child_args", nargs=argparse.REMAINDER)
    return parser


def main() -> int:
    args = _make_parser().parse_args()
    if args.restart_cutoff_seconds <= 0.0 or not math.isfinite(args.restart_cutoff_seconds):
        raise ValueError("restart_cutoff_seconds must be finite and positive")
    if args.required_consecutive_passes <= 0:
        raise ValueError("required_consecutive_passes must be positive")
    forwarded_args = list(args.child_args)
    if forwarded_args[:1] == ["--"]:
        forwarded_args.pop(0)
    _validate_forwarded_args(forwarded_args)

    trials: list[TrialOutcome] = []
    args.run_directory.mkdir(parents=True, exist_ok=True)
    for trial_index, train_seed in enumerate(args.train_seeds):
        gate_seed = args.gate_seeds[trial_index % len(args.gate_seeds)]
        trial = _run_trial(
            task=args.task,
            trial_index=trial_index,
            train_seed=train_seed,
            gate_seed=gate_seed,
            cutoff_seconds=args.restart_cutoff_seconds,
            required_consecutive_passes=args.required_consecutive_passes,
            run_directory=args.run_directory,
            forwarded_args=forwarded_args,
        )
        trials.append(trial)
        payload = {
            "schema": "phoenx_time_to_policy_v1",
            "task": args.task,
            "restart_policy": {
                "cutoff_seconds": args.restart_cutoff_seconds,
                "required_consecutive_passes": args.required_consecutive_passes,
            },
            "seed_protocol": {
                "train_seeds": list(args.train_seeds),
                "gate_seeds": list(args.gate_seeds),
                "pairing": "gate seeds cycle over ordered training seeds",
            },
            "host": {"hostname": platform.node(), "platform": platform.platform(), "python": sys.version},
            "command": [sys.executable, *sys.argv],
            "forwarded_child_args": forwarded_args,
            "trials": [asdict(item) for item in trials],
            "summary": summarize_trials(
                trials,
                args.restart_cutoff_seconds,
                bootstrap_samples=args.bootstrap_samples,
                bootstrap_seed=args.bootstrap_seed,
            ),
        }
        _write_json(args.output, payload)
        progress = {"latest_trial": asdict(trial), "summary": payload["summary"]}
        print(json.dumps(_json_compatible(progress), sort_keys=True, allow_nan=False))
    return 0 if all(trial.pass_gate for trial in trials) else 1


if __name__ == "__main__":
    raise SystemExit(main())
