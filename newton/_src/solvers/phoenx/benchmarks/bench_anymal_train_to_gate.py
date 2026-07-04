# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Measure PhoenX ANYmal curriculum time to a confirmed walking policy."""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import newton.rl as rl
from newton._src.solvers.phoenx.rl_training.examples import train_anymal_walk_phoenx_ppo as anymal


def benchmark_train_to_gate(args: argparse.Namespace) -> dict[str, Any]:
    if args.required_consecutive_passes <= 0:
        raise ValueError("required_consecutive_passes must be positive")
    if str(args.recipe) == "single":
        raise ValueError("time-to-policy requires a curriculum recipe")

    phases = anymal.anymal_recipe(str(args.recipe))
    selected = phases[int(args.start_phase) :]
    if args.phase_count is not None:
        selected = selected[: int(args.phase_count)]
    if not selected:
        raise ValueError("selected curriculum is empty")

    total_t0 = time.perf_counter()
    # Intermediate gates are diagnostics. Only the final frozen task gate
    # determines whether the resulting policy works.
    args.allow_gate_failure = True
    curriculum_result = anymal.train_curriculum(args)
    final_checkpoint = curriculum_result.get("final_checkpoint")
    gate_history: list[dict[str, Any]] = []
    first_pass: dict[str, Any] | None = None
    pass_streak = 0

    if final_checkpoint is not None:
        trainer = rl.load_ppo_checkpoint(
            final_checkpoint,
            config=anymal.build_ppo_config(args),
            device=args.device,
        )
        final_phase = selected[-1]
        for window in range(int(args.required_consecutive_passes)):
            gate_seed = int(args.gate_seed) + window * 1_000_003
            env_overrides = dict(final_phase.env_overrides)
            # The final forward phase contains stochastic pushes. A distinct
            # held-out seed therefore changes every confirmation window.
            env_overrides["disturbance_seed"] = gate_seed
            stats = anymal.evaluate_phase_commands(trainer, args, final_phase, env_overrides)
            failures = anymal.check_phase_gates(stats, final_phase)
            passed = not failures
            pass_streak = pass_streak + 1 if passed else 0
            entry = {
                "window": window,
                "gate_seed": gate_seed,
                "stats": [asdict(item) for item in stats],
                "failures": failures,
                "pass_gate": passed,
                "consecutive_passes": pass_streak,
                "qualified": pass_streak >= int(args.required_consecutive_passes),
                "total_wall_seconds": time.perf_counter() - total_t0,
            }
            gate_history.append(entry)
            if entry["qualified"]:
                first_pass = entry
                break
            if not passed:
                break

    result = {
        "engine": "phoenx_anymal_warp_ppo_train_to_gate",
        "metric": "fresh curriculum plus consecutive held-out disturbed walking gates",
        "recipe": str(args.recipe),
        "seed": int(args.seed),
        "gate_seed": int(args.gate_seed),
        "required_consecutive_passes": int(args.required_consecutive_passes),
        "pass_gate": first_pass is not None,
        "first_pass": first_pass,
        "gate_history": gate_history,
        "curriculum": curriculum_result,
        "checkpoint_path": final_checkpoint,
        "total_wall_seconds": time.perf_counter() - total_t0,
    }
    output_path = Path(args.json_output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, sort_keys=True))
    return result


def _make_parser() -> argparse.ArgumentParser:
    parser = anymal._make_parser()
    parser.description = __doc__
    parser.add_argument("--gate-seed", type=int, default=1000)
    parser.add_argument("--required-consecutive-passes", type=int, default=2)
    parser.add_argument("--json-output", required=True)
    return parser


def main() -> int:
    args = _make_parser().parse_args()
    result = benchmark_train_to_gate(args)
    return 0 if result["pass_gate"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
