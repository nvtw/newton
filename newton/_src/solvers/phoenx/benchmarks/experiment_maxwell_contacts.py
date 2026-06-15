# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import csv
import math
import os
import re
import subprocess
import sys
import time
from pathlib import Path


EXAMPLE_MODULE = "newton._src.solvers.phoenx.examples.example_maxwell_top_si"
MAXWELL_TOP_USD_ENV = "NEWTON_MAXWELL_TOP_USD"
DEFAULT_USD_PATH = Path(r"C:\Users\twidmer\Downloads\MaxwellTopSI\MaxwellTopSI2.usda")
BENCH_SOLVER = "phoenx"
BENCH_RIGID_CONTACT_MAX = 1000
BENCH_SDF_LINEARITY_BAD_RELERR = 0.2
TRAJECTORY_COMPARE_BASELINE = "gap0_sticky"


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[5]


def _tail_mean(rows: list[dict[str, str]], key: str, tail: int) -> float:
    vals = []
    for row in rows[-tail:]:
        value = row.get(key, "")
        if value == "":
            continue
        x = float(value)
        if math.isfinite(x):
            vals.append(x)
    if not vals:
        return float("nan")
    return sum(vals) / len(vals)


def _example_runner_source(
    args: argparse.Namespace,
    case: dict[str, str | float | bool],
    csv_path: Path,
    trajectory_path: Path,
) -> str:
    assignments = {
        "SOLVER_TYPE": BENCH_SOLVER,
        "CONTACT_GAP": float(case["gap"]),
        "CONTACT_MARGIN": float(case.get("margin", 0.0)),
        "RIGID_CONTACT_MAX": BENCH_RIGID_CONTACT_MAX,
        "REDUCE_CONTACTS": bool(case.get("reduce_contacts", True)),
        "CONTACT_MATCHING": str(case["matching"]),
        "SDF_CONTACT_SURFACE_FILTER_VOXELS": float(case.get("surface_filter_voxels", 0.0)),
        "CONTACT_REDUCTION_EXPORT_INNER_ONLY": bool(case.get("export_inner_only", False)),
        "CONTACT_STATS": True,
        "CONTACT_STATS_INTERVAL": int(args.interval),
        "CONTACT_STATS_CSV": str(csv_path),
        "TRAJECTORY_CSV": str(trajectory_path),
        "TRAJECTORY_INTERVAL": 1,
        "SDF_LINEARITY_BAD_RELERR": BENCH_SDF_LINEARITY_BAD_RELERR,
    }
    lines = [
        "import importlib",
        "import newton.examples",
        f"example = importlib.import_module({EXAMPLE_MODULE!r})",
    ]
    lines.extend(f"example.{name} = {value!r}" for name, value in assignments.items())
    lines.extend(
        [
            "viewer, args = newton.examples.init(example.Example.create_parser())",
            "newton.examples.run(example.Example(viewer, args), args)",
        ]
    )
    return "\n".join(lines)


def _run_case(args: argparse.Namespace, case: dict[str, str | float | bool]) -> dict[str, float | str]:
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / f"{case['name']}.csv"
    if csv_path.exists():
        csv_path.unlink()
    trajectory_path = out_dir / f"{case['name']}_trajectory.csv"
    if trajectory_path.exists():
        trajectory_path.unlink()

    cmd = [
        sys.executable,
        "-c",
        _example_runner_source(args, case, csv_path, trajectory_path),
        "--viewer",
        "null",
        "--num-frames",
        str(args.frames),
        "--quiet",
        "--usd-path",
        str(args.usd_path),
    ]

    print(f"\n=== {case['name']} ===")
    print(" ".join([cmd[0], cmd[1], "<example globals>", *cmd[3:]]))
    start_time = time.perf_counter()
    result = subprocess.run(
        cmd,
        cwd=_repo_root(),
        text=True,
        capture_output=True,
        timeout=args.timeout,
        check=False,
    )
    wall_time_s = time.perf_counter() - start_time
    if result.returncode != 0:
        print(result.stdout[-4000:])
        print(result.stderr[-4000:])
        raise RuntimeError(f"case {case['name']} failed with exit code {result.returncode}")

    with csv_path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise RuntimeError(f"case {case['name']} wrote no stats rows")

    last = rows[-1]
    summary: dict[str, float | str] = {
        "name": str(case["name"]),
        "gap": float(case["gap"]),
        "margin": float(case.get("margin", 0.0)),
        "matching": str(case["matching"]),
        "reduce_contacts": str(bool(case.get("reduce_contacts", True))),
        "surface_filter_voxels": str(case.get("surface_filter_voxels", 0.0)),
        "export_inner_only": str(bool(case.get("export_inner_only", False))),
        "trajectory_csv": str(trajectory_path),
        "wall_time_s": wall_time_s,
        "last_frame": float(last["frame"]),
        "tail_contacts": _tail_mean(rows, "contacts", args.tail),
        "tail_zero_gap": _tail_mean(rows, "zero_gap", args.tail),
        "tail_speculative": _tail_mean(rows, "speculative", args.tail),
        "tail_median_sep": _tail_mean(rows, "median_sep", args.tail),
        "tail_max_sep": _tail_mean(rows, "max_sep", args.tail),
        "tail_top_z": _tail_mean(rows, "top_z", args.tail),
        "tail_top_ang_speed": _tail_mean(rows, "top_ang_speed", args.tail),
        "tail_reducer_candidates": _tail_mean(rows, "reducer_candidates", args.tail),
        "tail_reducer_ht_active": _tail_mean(rows, "reducer_ht_active", args.tail),
        "tail_reducer_ht_insert_failures": _tail_mean(rows, "reducer_ht_insert_failures", args.tail),
        "tail_edge_anchor_p50": _tail_mean(rows, "edge_anchor_p50", args.tail),
        "tail_edge_anchor_over_sep_p50": _tail_mean(rows, "edge_anchor_over_sep_p50", args.tail),
        "tail_sdf_endpoint_abs_p50": _tail_mean(rows, "sdf_endpoint_abs_p50", args.tail),
        "tail_sdf_endpoint_abs_p95": _tail_mean(rows, "sdf_endpoint_abs_p95", args.tail),
        "tail_sdf_line_error_p50": _tail_mean(rows, "sdf_line_error_p50", args.tail),
        "tail_sdf_line_error_p95": _tail_mean(rows, "sdf_line_error_p95", args.tail),
        "tail_sdf_min_signed_delta_p05": _tail_mean(rows, "sdf_min_signed_delta_p05", args.tail),
        "tail_sdf_nonmonotone_frac": _tail_mean(rows, "sdf_nonmonotone_frac", args.tail),
        "tail_sdf_gen_delta_ratio_p50": _tail_mean(rows, "sdf_gen_delta_ratio_p50", args.tail),
        "tail_sdf_gen_delta_ratio_p95": _tail_mean(rows, "sdf_gen_delta_ratio_p95", args.tail),
        "tail_sdf_gen_delta_relerr_p50": _tail_mean(rows, "sdf_gen_delta_relerr_p50", args.tail),
        "tail_sdf_gen_delta_relerr_p95": _tail_mean(rows, "sdf_gen_delta_relerr_p95", args.tail),
        "tail_sdf_gen_bad_frac": _tail_mean(rows, "sdf_gen_bad_frac", args.tail),
        "tail_sdf_two_sided_delta_relerr_p50": _tail_mean(rows, "sdf_two_sided_delta_relerr_p50", args.tail),
        "tail_sdf_two_sided_delta_relerr_p95": _tail_mean(rows, "sdf_two_sided_delta_relerr_p95", args.tail),
        "tail_sdf_two_sided_bad_frac": _tail_mean(rows, "sdf_two_sided_bad_frac", args.tail),
    }
    print(
        "summary "
        f"contacts={summary['tail_contacts']:.3g} "
        f"zero_gap={summary['tail_zero_gap']:.3g} "
        f"speculative={summary['tail_speculative']:.3g} "
        f"median_sep={summary['tail_median_sep']:.6g} "
        f"max_sep={summary['tail_max_sep']:.6g} "
        f"top_z={summary['tail_top_z']:.6g} "
        f"edge_anchor_ratio={summary['tail_edge_anchor_over_sep_p50']:.6g} "
        f"sdf_endpoint={summary['tail_sdf_endpoint_abs_p50']:.6g} "
        f"sdf_line_error={summary['tail_sdf_line_error_p50']:.6g} "
        f"sdf_gen_bad={summary['tail_sdf_gen_bad_frac']:.6g} "
        f"sdf_two_bad={summary['tail_sdf_two_sided_bad_frac']:.6g}"
    )
    return summary


def _default_cases() -> list[dict[str, str | float | bool]]:
    return [
        {
            "name": "gap0_sticky",
            "gap": 0.0,
            "matching": "sticky",
            "reduce_contacts": True,
            "surface_filter_voxels": 1.0,
        },
        {
            "name": "gap0_margin001_sticky",
            "gap": 0.0,
            "margin": 0.001,
            "matching": "sticky",
            "reduce_contacts": True,
            "surface_filter_voxels": 1.0,
        },
        {
            "name": "gap0_margin01_sticky",
            "gap": 0.0,
            "margin": 0.01,
            "matching": "sticky",
            "reduce_contacts": True,
            "surface_filter_voxels": 1.0,
        },
        {
            "name": "gap001_surface_filter1_inner_export",
            "gap": 0.01,
            "matching": "sticky",
            "reduce_contacts": True,
            "surface_filter_voxels": 1.0,
            "export_inner_only": True,
        },
        {
            "name": "gap001_unfiltered_inner_export",
            "gap": 0.01,
            "matching": "sticky",
            "reduce_contacts": True,
            "surface_filter_voxels": 0.0,
            "export_inner_only": True,
        },
        {
            "name": "gap001_surface_filter1",
            "gap": 0.01,
            "matching": "sticky",
            "reduce_contacts": True,
            "surface_filter_voxels": 1.0,
        },
        {
            "name": "gap001_surface_filter2",
            "gap": 0.01,
            "matching": "sticky",
            "reduce_contacts": True,
            "surface_filter_voxels": 2.0,
        },
        {
            "name": "gap001_surface_filter4",
            "gap": 0.01,
            "matching": "sticky",
            "reduce_contacts": True,
            "surface_filter_voxels": 4.0,
        },
        {
            "name": "gap001_unfiltered",
            "gap": 0.01,
            "matching": "sticky",
            "reduce_contacts": True,
            "surface_filter_voxels": 0.0,
        },
        {
            "name": "gap001_unfiltered_latest",
            "gap": 0.01,
            "matching": "latest",
            "reduce_contacts": True,
            "surface_filter_voxels": 0.0,
        },
        {
            "name": "gap001_margin001_unfiltered",
            "gap": 0.01,
            "margin": 0.001,
            "matching": "sticky",
            "reduce_contacts": True,
            "surface_filter_voxels": 0.0,
        },
        {
            "name": "gap001_margin001_surface_filter1",
            "gap": 0.01,
            "margin": 0.001,
            "matching": "sticky",
            "reduce_contacts": True,
            "surface_filter_voxels": 1.0,
        },
        {
            "name": "gap001_margin01_unfiltered",
            "gap": 0.01,
            "margin": 0.01,
            "matching": "sticky",
            "reduce_contacts": True,
            "surface_filter_voxels": 0.0,
        },
        {
            "name": "gap001_margin01_surface_filter1",
            "gap": 0.01,
            "margin": 0.01,
            "matching": "sticky",
            "reduce_contacts": True,
            "surface_filter_voxels": 1.0,
        },
        {
            "name": "gap01_unfiltered",
            "gap": 0.1,
            "matching": "sticky",
            "reduce_contacts": True,
            "surface_filter_voxels": 0.0,
        },
        {
            "name": "gap01_unfiltered_latest",
            "gap": 0.1,
            "matching": "latest",
            "reduce_contacts": True,
            "surface_filter_voxels": 0.0,
        },
        {
            "name": "gap01_unfiltered_inner_export",
            "gap": 0.1,
            "matching": "sticky",
            "reduce_contacts": True,
            "surface_filter_voxels": 0.0,
            "export_inner_only": True,
        },
        {
            "name": "gap01_surface_filter1",
            "gap": 0.1,
            "matching": "sticky",
            "reduce_contacts": True,
            "surface_filter_voxels": 1.0,
        },
        {
            "name": "gap01_surface_filter1_inner_export",
            "gap": 0.1,
            "matching": "sticky",
            "reduce_contacts": True,
            "surface_filter_voxels": 1.0,
            "export_inner_only": True,
        },
        {
            "name": "gap01_surface_filter2",
            "gap": 0.1,
            "matching": "sticky",
            "reduce_contacts": True,
            "surface_filter_voxels": 2.0,
        },
        {
            "name": "gap01_surface_filter4",
            "gap": 0.1,
            "matching": "sticky",
            "reduce_contacts": True,
            "surface_filter_voxels": 4.0,
        },
        {
            "name": "gap01_no_reduce",
            "gap": 0.1,
            "matching": "sticky",
            "reduce_contacts": False,
            "surface_filter_voxels": 1.0,
        },
    ]


def _label_body_path(body_path: str) -> str:
    parts = [part for part in body_path.split("/") if part]
    label = parts[-2] if len(parts) >= 2 and parts[-1].endswith("_obj0") else parts[-1]
    return re.sub(r"[^A-Za-z0-9]+", "_", label).strip("_").lower()


def _load_trajectory(path: Path) -> dict[str, dict[int, tuple[float, float, float]]]:
    rows: dict[str, dict[int, tuple[float, float, float]]] = {}
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            body_path = row["body_path"]
            frame = int(row["frame"])
            rows.setdefault(body_path, {})[frame] = (
                float(row["tip_x"]),
                float(row["tip_y"]),
                float(row["tip_z"]),
            )
    return rows


def _trajectory_compare(ref_path: Path, case_path: Path) -> dict[str, float]:
    ref = _load_trajectory(ref_path)
    case = _load_trajectory(case_path)
    metrics: dict[str, float] = {}
    for body_path in sorted(set(ref) & set(case)):
        common_frames = sorted(set(ref[body_path]) & set(case[body_path]))
        if not common_frames:
            continue

        deltas = []
        dz_abs = []
        for frame in common_frames:
            r = ref[body_path][frame]
            c = case[body_path][frame]
            dx = c[0] - r[0]
            dy = c[1] - r[1]
            dz = c[2] - r[2]
            deltas.append(math.sqrt(dx * dx + dy * dy + dz * dz))
            dz_abs.append(abs(dz))

        label = _label_body_path(body_path)
        metrics[f"{label}_tip_delta_final"] = deltas[-1]
        metrics[f"{label}_tip_delta_max"] = max(deltas)
        metrics[f"{label}_tip_delta_rms"] = math.sqrt(sum(d * d for d in deltas) / len(deltas))
        metrics[f"{label}_tip_z_delta_abs_max"] = max(dz_abs)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames", type=int, default=80)
    parser.add_argument("--interval", type=int, default=5)
    parser.add_argument("--tail", type=int, default=6)
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument(
        "--usd-path",
        default=os.environ.get(MAXWELL_TOP_USD_ENV, str(DEFAULT_USD_PATH)),
        help=f"Path to MaxwellTopSI2.usda. Defaults to {MAXWELL_TOP_USD_ENV} or the historical local path.",
    )
    parser.add_argument("--output-dir", default=r"C:\tmp\maxwell_contact_experiment")
    parser.add_argument("--case", action="append", default=None, help="Run only matching case name(s).")
    args = parser.parse_args()

    cases = _default_cases()
    if args.case:
        selected = set(args.case)
        cases = [case for case in cases if str(case["name"]) in selected]
        missing = selected - {str(case["name"]) for case in cases}
        if missing:
            raise ValueError(f"unknown case(s): {', '.join(sorted(missing))}")

    summaries = [_run_case(args, case) for case in cases]
    baseline = next((summary for summary in summaries if summary["name"] == TRAJECTORY_COMPARE_BASELINE), None)
    if baseline is not None:
        baseline_path = Path(str(baseline["trajectory_csv"]))
        for summary in summaries:
            if summary is baseline:
                continue
            metrics = _trajectory_compare(baseline_path, Path(str(summary["trajectory_csv"])))
            summary.update(metrics)
            if metrics:
                metric_str = " ".join(f"{key}={value:.6g}" for key, value in sorted(metrics.items()))
                print(f"trajectory_vs_{TRAJECTORY_COMPARE_BASELINE} {summary['name']}: {metric_str}")

    out_path = Path(args.output_dir) / "summary.csv"
    with out_path.open("w", newline="") as f:
        fieldnames: list[str] = []
        for summary in summaries:
            for key in summary:
                if key not in fieldnames:
                    fieldnames.append(key)
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summaries)
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
